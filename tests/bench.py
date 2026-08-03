#!/usr/bin/env python3
"""Small repeatable benchmark for Alpaccaroo inference.

The harness itself lives in :mod:`alpaccaroo.bench` so `alpaccaroo bench`
and this script are the same code; this file stays as the in-repo entry
point the README and CI already call, and adds `--bench-lines` so its
output keeps the greppable `BENCH key=value` form.

    python3 tests/bench.py --model qwen3bmed
    python3 tests/bench.py --model qwen3bmed --profile-json perf.json
    python3 tests/bench.py --model /tmp/b1.gguf --prefill 32 --decode 8

Model references resolve locally first - nicknames and installed names
included - and a reference that is not installed is an error rather than a
download, unless --allow-pull is given.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from alpaccaroo.bench import main as _main  # noqa: E402


def main() -> int:
    argv = list(sys.argv[1:])
    if "--bench-lines" not in argv:
        argv.append("--bench-lines")
    return _main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
