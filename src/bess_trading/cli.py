"""Command-line entry points for reproducible data collection."""

from __future__ import annotations

import argparse
import logging

from bess_trading.data.entsoe import EntsoeClient, save_raw
from bess_trading.data.quality import validate_market_frame


def main() -> None:
    parser = argparse.ArgumentParser(prog="bess-research")
    parser.add_argument("--log-level", default="INFO")
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch-prices", help="Download ENTSO-E day-ahead prices")
    fetch.add_argument("--start", required=True, help="ISO timestamp, interpreted as UTC if naive")
    fetch.add_argument("--end", required=True)
    fetch.add_argument("--output", default="data/raw/entsoe_day_ahead.parquet")
    args = parser.parse_args()
    logging.basicConfig(
        level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    if args.command == "fetch-prices":
        frame = EntsoeClient().day_ahead_prices(args.start, args.end)
        report = validate_market_frame(frame)
        report.raise_if_invalid()
        save_raw(frame, args.output)


if __name__ == "__main__":
    main()
