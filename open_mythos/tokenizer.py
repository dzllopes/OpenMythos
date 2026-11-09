from transformers import AutoTokenizer

DEFAULT_MODEL_ID = "openai/gpt-oss-20b"
# Cadeia de suprimentos: fixa o commit do repо no Hub, para não baixar conteúdo
# alterado upstream (conta comprometida, rename, etc.). Resolvido em 2026-11-09;
# atualize junto com DEFAULT_MODEL_ID. Vale só para o modelo padrão — um model_id
# customizado usa a última revisão (revision=None), como antes.
DEFAULT_REVISION = "6cee5e81ee83917806bbde320786a8fb61efebee"


class MythosTokenizer:
    """
    HuggingFace tokenizer wrapper for OpenMythos.

    Args:
        model_id (str): The HuggingFace model ID or path to use with AutoTokenizer.
            Defaults to "openai/gpt-oss-20b".

    Attributes:
        tokenizer: An instance of HuggingFace's AutoTokenizer.

    Example:
        >>> tok = MythosTokenizer()
        >>> ids = tok.encode("Hello world")
        >>> s = tok.decode(ids)
    """

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, revision: str | None = None):
        """
        Initialize the MythosTokenizer.

        Args:
            model_id (str): HuggingFace model identifier or path to tokenizer files.
            revision (str | None): Hub revision (commit SHA/tag) to pin. When None
                and model_id is the default, pins DEFAULT_REVISION for supply-chain
                safety; for a custom model_id, None means the latest revision.
        """
        if revision is None and model_id == DEFAULT_MODEL_ID:
            revision = DEFAULT_REVISION
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

    @property
    def vocab_size(self) -> int:
        """
        Return the size of the tokenizer vocabulary.

        Returns:
            int: The number of unique tokens in the tokenizer vocabulary.
        """
        return self.tokenizer.vocab_size

    def encode(self, text: str) -> list[int]:
        """
        Encode input text into a list of token IDs.

        Args:
            text (str): The input text string to tokenize.

        Returns:
            list[int]: List of integer token IDs representing the input text.
        """
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, token_ids: list[int]) -> str:
        """
        Decode a list of token IDs back into a text string.

        Args:
            token_ids (list[int]): A list of integer token IDs to decode.

        Returns:
            str: Decoded string representation of the token IDs.
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
