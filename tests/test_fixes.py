import torch
import torch.nn.functional as F

from open_mythos.main import MoEFFN, MythosConfig, OpenMythos, RecurrentBlock


def small_cfg(**kw):
    base = dict(
        vocab_size=100,
        dim=64,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=32,
        max_loop_iters=4,
        prelude_layers=1,
        coda_layers=1,
        n_experts=8,
        n_shared_experts=1,
        n_experts_per_tok=2,
        expert_dim=16,
        lora_rank=4,
        attn_type="gqa",
    )
    base.update(kw)
    return MythosConfig(**base)


def test_star_import_works():
    ns = {}
    exec("from open_mythos import *", ns)
    for name in ("OpenMythos", "MythosTokenizer", "MoDAConfig", "MoDAModel"):
        assert name in ns


def _reference_moe(moe, x):
    """Original top-K × n_experts masked dispatch, kept as an oracle."""
    B, T, D = x.shape
    flat = x.view(B * T, D)
    logits = moe.router(flat)
    scores = F.softmax(logits, dim=-1)
    _, topk_idx = (logits + moe.router_bias).topk(moe.topk, dim=-1)
    topk_scores = scores.gather(-1, topk_idx)
    topk_scores = topk_scores / topk_scores.sum(dim=-1, keepdim=True)
    out = torch.zeros_like(flat)
    for i in range(moe.topk):
        ids = topk_idx[:, i]
        s = topk_scores[:, i].unsqueeze(-1)
        for eid in range(moe.n_experts):
            m = ids == eid
            if m.any():
                out[m] += s[m] * moe.routed_experts[eid](flat[m])
    for shared in moe.shared_experts:
        out = out + shared(flat)
    return out.view(B, T, D)


def test_moe_sorted_dispatch_matches_reference():
    torch.manual_seed(0)
    moe = MoEFFN(small_cfg()).eval()
    x = torch.randn(3, 7, 64)
    torch.testing.assert_close(moe(x), _reference_moe(moe, x), rtol=1e-5, atol=1e-5)


def test_moe_counts_load_only_in_training():
    moe = MoEFFN(small_cfg())
    x = torch.randn(2, 5, 64)
    moe.eval()
    moe(x)
    assert moe.expert_load.sum() == 0
    moe.train()
    moe(x)
    assert moe.expert_load.sum() == 2 * 5 * moe.topk


def test_router_bias_update_favors_underused_experts():
    moe = MoEFFN(small_cfg())
    moe.expert_load.copy_(torch.tensor([10.0, 0, 0, 0, 0, 0, 0, 6]))
    moe.update_router_bias(rate=0.01)
    assert moe.router_bias[0] < 0  # overloaded → lowered
    assert moe.router_bias[1] > 0  # unused → raised
    assert moe.expert_load.sum() == 0  # counts reset


def test_router_bias_balances_skewed_router():
    torch.manual_seed(0)
    moe = MoEFFN(small_cfg()).train()
    with torch.no_grad():
        moe.router.weight.zero_()
        moe.router.weight[0] += 0.02  # experts 0 and 1 preferred by every token
        moe.router.weight[1] += 0.02
    x = torch.randn(4, 16, 64).abs()

    def max_share():
        moe.expert_load.zero_()
        moe(x)
        load = moe.expert_load.clone()
        moe.expert_load.zero_()
        return (load / load.sum()).max().item()

    before = max_share()
    for _ in range(300):
        moe(x)
        moe.update_router_bias(rate=0.01)
    after = max_share()
    assert after < before


def test_model_update_router_biases_touches_all_moe_layers():
    model = OpenMythos(small_cfg()).train()
    ids = torch.randint(0, 100, (2, 8))
    model(ids, n_loops=2)
    model.update_router_biases(1e-3)
    moes = [m for m in model.modules() if isinstance(m, MoEFFN)]
    assert moes and all(m.expert_load.sum() == 0 for m in moes)
    assert any(m.router_bias.abs().sum() > 0 for m in moes)


def test_act_weights_sum_to_one_when_loops_run_out():
    """With 1 loop nothing can cross the threshold; the output must still be h, not p·h."""
    torch.manual_seed(0)
    cfg = small_cfg()
    rb = RecurrentBlock(cfg).eval()
    B, T = 2, 6
    h = torch.randn(B, T, cfg.dim)
    e = torch.randn(B, T, cfg.dim)
    from open_mythos.main import precompute_rope_freqs

    freqs = precompute_rope_freqs(cfg.dim // cfg.n_heads, cfg.max_seq_len)[:T]
    out = rb(h, e, freqs, None, n_loops=1)

    # recompute the single-step hidden state the block produced
    from open_mythos.main import loop_index_embedding

    h_loop = loop_index_embedding(h, 0, rb.loop_dim)
    trans = rb.block(rb.norm(h_loop + e), freqs, None, None, "recurrent_loop_0")
    trans = trans + rb.lora(trans, 0)
    h1 = rb.injection(h, e, trans)
    torch.testing.assert_close(out, h1, rtol=1e-5, atol=1e-5)
