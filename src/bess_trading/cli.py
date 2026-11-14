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
    fundamentals = sub.add_parser(
        "fetch-fundamentals", help="Download archived day-ahead load, wind and solar forecasts"
    )
    fundamentals.add_argument("--start", required=True)
    fundamentals.add_argument("--end", required=True)
    fundamentals.add_argument("--output-dir", default="data/raw")
    run_phase1 = sub.add_parser("run-phase1", help="Run the audited Phase 1 experiment")
    run_phase1.add_argument("--config", default="configs/phase1.yaml")
    run_phase2 = sub.add_parser("run-phase2", help="Run the probabilistic Phase 2 experiment")
    run_phase2.add_argument("--config", default="configs/phase2.yaml")
    run_kumo = sub.add_parser(
        "run-kumo-diagnostic", help="Run the pre-test Kumo research feasibility diagnostic"
    )
    run_kumo.add_argument("--config", default="configs/kumo_research.yaml")
    run_kumo_expanded = sub.add_parser(
        "run-kumo-expanded", help="Run the frozen 24-origin Kumo development benchmark"
    )
    run_kumo_expanded.add_argument("--config", default="configs/kumo_development_24.yaml")
    run_kumo_conditional = sub.add_parser(
        "run-kumo-conditional", help="Run the pre-registered conditional Kumo study"
    )
    run_kumo_conditional.add_argument(
        "--config", default="configs/kumo_conditional_research.yaml"
    )
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
    elif args.command == "fetch-fundamentals":
        client = EntsoeClient()
        output_dir = Path(args.output_dir)
        for start, end in annual_chunks(args.start, args.end):
            load = client.day_ahead_load_forecast(start, end)
            renewables = client.day_ahead_wind_solar_forecast(start, end)
            frame = load.join(renewables, how="outer").sort_index()
            target = output_dir / f"entsoe_fundamentals_{start:%Y%m%d}_{end:%Y%m%d}.parquet"
            save_raw(frame, target)
    elif args.command == "run-phase1":
        from bess_trading.experiments.phase1 import run

        result = run(args.config)
        logging.getLogger(__name__).info(
            "Phase 1 completed with %s delivery days",
            result["experiment"]["complete_dispatch_days"],
        )
    elif args.command == "run-phase2":
        from bess_trading.experiments.phase2 import run

        result = run(args.config)
        logging.getLogger(__name__).info(
            "Phase 2 completed with %s delivery days", result["experiment"]["test_days"]
        )
    elif args.command == "run-kumo-diagnostic":
        from bess_trading.experiments.kumo_benchmark import run

        result = run(args.config)
        logging.getLogger(__name__).info(
            "Kumo development gate: %s",
            result["research_feasibility_gate"]["proceed_to_full_development"],
        )
    elif args.command == "run-kumo-expanded":
        from bess_trading.experiments.kumo_expanded_benchmark import run

        result = run(args.config)
        logging.getLogger(__name__).info(
            "Expanded Kumo research gate: %s", result["research_gate"]["decision"]
        )
    elif args.command == "run-kumo-conditional":
        from bess_trading.experiments.kumo_conditional_benchmark import run

        result = run(args.config)
        logging.getLogger(__name__).info(
            "Conditional Kumo supported: %s",
            result["decision"]["conditional_policy_supported"],
        )


if __name__ == "__main__":
    main()
