"""Pre-deployment scanners.  Five checks, one CLI, ninety seconds on stage.

    uv run python -m attacklab.supplychain --all

Demo 3's script is four results, and this package produces them in that order:

1. the pickle-format model **flagged** and its safetensors twin **passed** --
   same weights, different container;
2. the poisoned document from demo 1 found by **content**, with the manifest
   flag unread -- a hash tells you the corpus *changed*, not that what changed
   was hostile;
3. one skill manifest requesting a tool it cannot justify;
4. one moved digest, so the lockfile check fails.

The fifth, :mod:`~attacklab.supplychain.scan_tool_surface`, is the highest
value-per-line check in the repository and does not need a fixture: it diffs the
functions actually exposed to the model against a reviewed allowlist.
"""

__all__ = [
    "scan_index",
    "scan_lock",
    "scan_model",
    "scan_skills",
    "scan_tool_surface",
]
