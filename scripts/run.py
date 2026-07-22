#!/usr/bin/env python
"""Cross-platform, make-free entrypoint.

    python scripts/run.py corpus     # regenerate the frozen corpus
    python scripts/run.py queries    # regenerate the frozen query sets
    python scripts/run.py eval       # run the full evaluation
    python scripts/run.py figures    # regenerate paper figures
    python scripts/run.py test       # run the test suite
    python scripts/run.py all        # corpus -> queries -> eval -> figures

Every subcommand is a thin shim over the same module `make` targets invoke, so
Windows users without `make` reproduce the paper identically.
"""

from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(args: list[str]) -> int:
    print("+ " + " ".join(args))
    return subprocess.call([sys.executable, *args], cwd=ROOT)


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "eval"
    extra = sys.argv[2:]
    steps: dict[str, list[list[str]]] = {
        "corpus": [["-m", "corpus.generate", "--seed", "42"]],
        "queries": [["-m", "harness.build_queries", "--seed", "1234"]],
        "eval": [["-m", "harness.run_eval", *extra]],
        "figures": [["-m", "paper.generate_figures"]],
        "pdf": [["-m", "paper.generate_figures"], ["-m", "paper.build_pdf"]],
        "test": [["-m", "pytest", "tests/", "-q"]],
        "all": [
            ["-m", "corpus.generate", "--seed", "42"],
            ["-m", "harness.build_queries", "--seed", "1234"],
            ["-m", "harness.run_eval", *extra],
            ["-m", "paper.generate_figures"],
        ],
    }
    if cmd not in steps:
        print(f"unknown command: {cmd!r}\nvalid: {', '.join(steps)}")
        return 2
    for step in steps[cmd]:
        rc = _run(step)
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
