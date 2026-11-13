"""Compatibility wrapper for the configuration-driven Phase 1 benchmark."""

from __future__ import annotations

import argparse
import json

from bess_trading.experiments.phase1 import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase1.yaml")
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2))


if __name__ == "__main__":
    main()
