"""Current German energy and balancing product contracts.

Rules verified against regelleistung.net on 2026-11-13. Keep the verification date
visible: market rules are versioned inputs, not timeless constants.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from enum import StrEnum

import pandas as pd


class Market(StrEnum):
    DAY_AHEAD = "day_ahead"
    INTRADAY = "intraday"
    FCR = "fcr"
    AFRR_CAPACITY = "afrr_capacity"
    AFRR_ENERGY = "afrr_energy"
    MFRR_CAPACITY = "mfrr_capacity"
    MFRR_ENERGY = "mfrr_energy"


@dataclass(frozen=True)
class ProductRule:
    market: Market
    resolution_minutes: int
    block_hours: int | None
    gate_closure_day_offset: int
    gate_closure_local: time | None
    gate_closure_minutes_before: int | None
    minimum_bid_mw: float
    bid_increment_mw: float
    symmetric: bool
    remuneration: str
    verified_on: str = "2026-11-13"


PRODUCT_RULES = {
    Market.DAY_AHEAD: ProductRule(
        Market.DAY_AHEAD, 15, None, -1, time(12, 0), None, 0.1, 0.1, False, "pay_as_clear"
    ),
    Market.INTRADAY: ProductRule(
        Market.INTRADAY, 15, None, 0, None, 5, 0.1, 0.1, False, "continuous_trade"
    ),
    Market.FCR: ProductRule(
        Market.FCR, 15, 4, -1, time(8, 0), None, 1.0, 1.0, True, "pay_as_clear"
    ),
    Market.AFRR_CAPACITY: ProductRule(
        Market.AFRR_CAPACITY, 15, 4, -1, time(9, 0), None, 1.0, 1.0, False, "pay_as_bid"
    ),
    Market.AFRR_ENERGY: ProductRule(
        Market.AFRR_ENERGY, 15, None, 0, None, 25, 1.0, 1.0, False, "pay_as_clear"
    ),
    Market.MFRR_CAPACITY: ProductRule(
        Market.MFRR_CAPACITY, 15, 4, -1, time(10, 0), None, 1.0, 1.0, False, "pay_as_bid"
    ),
    Market.MFRR_ENERGY: ProductRule(
        Market.MFRR_ENERGY, 15, None, 0, None, 25, 1.0, 1.0, False, "pay_as_clear"
    ),
}


def gate_closure(delivery: pd.Timestamp, market: Market) -> pd.Timestamp:
    """Return the Europe/Berlin gate closure converted to UTC."""
    rule = PRODUCT_RULES[market]
    local = delivery.tz_convert("Europe/Berlin")
    if rule.gate_closure_minutes_before is not None:
        return delivery - pd.Timedelta(minutes=rule.gate_closure_minutes_before)
    assert rule.gate_closure_local is not None
    day = local.normalize() + pd.Timedelta(days=rule.gate_closure_day_offset)
    close = day + pd.Timedelta(
        hours=rule.gate_closure_local.hour, minutes=rule.gate_closure_local.minute
    )
    return close.tz_convert("UTC")


def four_hour_block(index: pd.DatetimeIndex) -> pd.Series:
    """Return DST-safe local capacity-product block identifiers."""
    local = index.tz_convert("Europe/Berlin")
    labels = [f"{stamp.date()}_{stamp.hour // 4:02d}" for stamp in local]
    return pd.Series(labels, index=index, name="capacity_block")
