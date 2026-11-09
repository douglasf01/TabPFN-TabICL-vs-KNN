"""Comparing tabular foundation models (TabPFN, TabICL) with KNN."""

import os
from pathlib import Path


def _load_env():
    """Read KEY=VALUE lines from the project's .env into os.environ (no override)."""
    env = Path(__file__).resolve().parent.parent / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.removeprefix("export ").split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))
    # TabPFN reads TABPFN_TOKEN; also accept TAB_PFN_TOKEN.
    if "TAB_PFN_TOKEN" in os.environ:
        os.environ.setdefault("TABPFN_TOKEN", os.environ["TAB_PFN_TOKEN"])


_load_env()
