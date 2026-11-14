from __future__ import annotations

from pathlib import Path

import pandas as pd

from bess_trading.config import load_yaml


def test_frozen_expanded_panel_matches_manifest_and_excludes_native_darr() -> None:
    settings = load_yaml("configs/kumo_development_24.yaml")
    configured = settings["experiment"]["development_dates"]
    manifest = pd.read_csv("configs/kumo_development_origins_v1.csv")
    assert 20 <= len(configured) <= 30
    assert configured == sorted(configured)
    assert configured == manifest.delivery_day.tolist()
    assert settings["native_darr"]["enabled"] is False
    assert pd.Timestamp(max(configured), tz="Europe/Berlin").tz_convert("UTC") < pd.Timestamp(
        settings["experiment"]["retrospective_test_start"]
    )


def test_expanded_artifacts_do_not_overwrite_frozen_benchmarks() -> None:
    settings = load_yaml("configs/kumo_development_24.yaml")
    target = Path(settings["experiment"]["output"])
    assert target == Path("results/kumo_development_24")
    assert target not in {Path("results/phase1_2024"), Path("results/phase2_2024")}
