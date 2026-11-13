"""Command-line entry points for reproducible data collection."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from bess_trading.data.entsoe import EntsoeClient, save_raw
from bess_trading.data.quality import validate_market_frame
from bess_trading.data.ranges import annual_chunks


def main() -> None:
    parser = argparse.ArgumentParser(prog="bess-research")
    parser.add_argument("--log-level", default="INFO")
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch-prices", help="Download ENTSO-E day-ahead prices")
    fetch.add_argument("--start", required=True, help="ISO timestamp, interpreted as UTC if naive")
    fetch.add_argument("--end", required=True)
    fetch.add_argument("--output", default="data/raw/entsoe_day_ahead.parquet")
    history = sub.add_parser("fetch-history", help="Download multi-year prices in annual chunks")
    history.add_argument("--start", required=True)
    history.add_argument("--end", required=True)
    history.add_argument("--output-dir", default="data/raw")
    run_phase1 = sub.add_parser("run-phase1", help="Run the audited Phase 1 experiment")
    run_phase1.add_argument("--config", default="configs/phase1.yaml")
    args = parser.parse_args()
    logging.basicConfig(
        level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    if args.command == "fetch-prices":
        frame = EntsoeClient().day_ahead_prices(args.start, args.end)
        report = validate_market_frame(frame)
        report.raise_if_invalid()
        save_raw(frame, args.output)
    elif args.command == "fetch-history":
        client = EntsoeClient()
        output_dir = Path(args.output_dir)
        for start, end in annual_chunks(args.start, args.end):
            frame = client.day_ahead_prices(start, end)
            report = validate_market_frame(frame)
            report.raise_if_invalid()
            target = output_dir / f"entsoe_prices_{start:%Y%m%d}_{end:%Y%m%d}.parquet"
            save_raw(frame, target)
    elif args.command == "run-phase1":
        from bess_trading.experiments.phase1 import run

        result = run(args.config)
        logging.getLogger(__name__).info(
            "Phase 1 completed with %s delivery days",
            result["experiment"]["complete_dispatch_days"],
        )


if __name__ == "__main__":
    main()
