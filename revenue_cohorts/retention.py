"""Net revenue retention by size tier: the cohort present with nonzero
spend at the start of the trailing window, measured again at the end of
the window. New logos that joined during the window are excluded, which is
the standard NRR definition and the reason this is not simply
end-of-window total revenue over start-of-window total revenue.
"""
from __future__ import annotations

from . import config


def compute_nrr(panel, trailing_months: int = config.TRAILING_MONTHS) -> dict[str, float]:
    end_month = panel["month"].max()
    start_month = end_month - trailing_months
    start_df = panel.loc[panel["month"] == start_month, ["advertiser_id", "size_tier", "spend"]]
    end_df = panel.loc[panel["month"] == end_month, ["advertiser_id", "spend"]]
    merged = start_df.rename(columns={"spend": "spend_start"}).merge(
        end_df.rename(columns={"spend": "spend_end"}), on="advertiser_id"
    )
    cohort = merged.loc[merged["spend_start"] > 0]

    out = {}
    for tier in config.SIZE_TIERS:
        sub = cohort.loc[cohort["size_tier"] == tier]
        start_total = sub["spend_start"].sum()
        out[tier] = float(sub["spend_end"].sum() / start_total * 100.0) if start_total > 0 else float("nan")
    return out
