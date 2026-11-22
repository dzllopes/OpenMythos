import os
import sys

# allow running from a source checkout without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from open_mythos import (
    mythos_1b,
    OpenMythos,
)

cfg = mythos_1b()
model = OpenMythos(cfg)

total = sum(p.numel() for p in model.parameters())
print(f"Parameters: {total:,}")
