"""Decomposes month-over-month spend change into new, expansion, contraction
and churn dollars, the standard SaaS/ad-revenue bridge. Every advertiser
falls into exactly one bucket each month (or none, if spend was 0 in both
months), which is checked directly in tests rather than assumed.
"""
from __future__ import annotations

import pandas as pd

from . import config


def monthly_bridge(panel: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    """Returns one row per month (optionally x each value of `by`, e.g.
    ["size_tier"]) with new_dollars, expansion_dollars, contraction_dollars
    and churn_dollars (all reported as positive magnitudes; churn and
    contraction are losses)."""
    panel = panel.sort_values(["advertiser_id", "month"]).copy()
    panel["prev_spend"] = panel.groupby("advertiser_id")["spend"].shift(1)
    scored = panel.loc[panel["month"] > 0].copy()

    new_mask = (scored["prev_spend"] == 0) & (scored["spend"] > 0)
    churn_mask = (scored["prev_spend"] > 0) & (scored["spend"] == 0)
    both_mask = (scored["prev_spend"] > 0) & (scored["spend"] > 0)
    expansion_mask = both_mask & (scored["spend"] > scored["prev_spend"])
    contraction_mask = both_mask & (scored["spend"] < scored["prev_spend"])

    scored["new_dollars"] = scored["spend"].where(new_mask, 0.0)
    scored["churn_dollars"] = scored["prev_spend"].where(churn_mask, 0.0)
    scored["expansion_dollars"] = (scored["spend"] - scored["prev_spend"]).where(expansion_mask, 0.0)
    scored["contraction_dollars"] = (scored["prev_spend"] - scored["spend"]).where(contraction_mask, 0.0)

    group_cols = ["month"] + (by or [])
    agg = scored.groupby(group_cols)[
        ["new_dollars", "expansion_dollars", "contraction_dollars", "churn_dollars"]
    ].sum().reset_index()
    return agg
