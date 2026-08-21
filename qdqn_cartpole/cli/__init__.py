"""Console entry points for qdqn-cartpole.

Installed as `qdqn-train`, `qdqn-sweep`, ... -- see [project.scripts].
"""
from __future__ import annotations

import os
import sys


def announce_paths(**dirs: str) -> None:
    """Print the resolved input/output directories on startup.

    Every script here addresses results/ and figures/ relative to the current
    working directory. Run as `python scripts/train.py` from the repo root that
    meant the repo's own results/; installed as `qdqn-train` and invoked from
    somewhere else it silently creates a *new* results/ in whatever directory
    the user happened to be in, and `qdqn-summarize` then reports "no runs
    found" against a repo full of data.

    The paths stay cwd-relative -- hard-coding an absolute path would paper over
    the problem rather than surface it -- so this makes a wrong cwd visible in
    the first line of the log instead of after the run.

    Written to stderr on purpose: stdout stays byte-identical to the
    pre-packaging scripts, which is what the summarize regression test diffs.
    """
    parts = "  ".join(f"{k}={os.path.abspath(v)}" for k, v in dirs.items())
    print(f"[qdqn] cwd={os.getcwd()}  {parts}", file=sys.stderr)
