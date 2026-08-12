"""Tiny tee-logger: writes every line to stdout AND a file, flushing both.

Both the terminal and the log file update live (line by line), so training can be
followed in real time with `tail -f results/{name}_{seed}.log` (Git Bash) or
`Get-Content results/{name}_{seed}.log -Wait` (PowerShell). The per-episode CSV
(written by the trainer) is likewise flushed every episode.
"""
from __future__ import annotations

import sys


class TeeLogger:
    def __init__(self, path: str):
        self.path = path
        self._f = open(path, "w", encoding="utf-8")

    def __call__(self, msg: str = "") -> None:
        text = str(msg)
        print(text, flush=True)
        self._f.write(text + "\n")
        self._f.flush()

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass

    def __enter__(self) -> "TeeLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
