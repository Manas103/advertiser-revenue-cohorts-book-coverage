"""Propensity-to-grow: a model that scores every advertiser on how likely
it is to be an "expander" (spend up, not new, not churned) in the next
quarter, trained on historical quarter-over-quarter transitions and
evaluated out of sample. Reported by its decision value, the share of
actual next-quarter expansion dollars captured by the top decile of
scores, rather than by AUC: the reader this project is written for buys
decisions, not model metrics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from . import config

QUARTER_MONTHS = 3
FEATURES = ["trailing_growth_rate", "tenure_months", "is_large"]


def _quarter_frame(panel: pd.DataFrame) -> pd.DataFrame:
    """One row per (advertiser_id, quarter) with spend at the quarter's
    first and last month, tenure in months at quarter start, and whether
    the advertiser became an expander that quarter."""
    n_quarters = config.N_MONTHS // QUARTER_MONTHS
    rows = []
    for q in range(n_quarters):
        start_m = q * QUARTER_MONTHS
        end_m = start_m + QUARTER_MONTHS - 1
        start_df = panel.loc[panel["month"] == start_m, ["advertiser_id", "size_tier", "owner_id", "spend", "start_month"]]
        end_df = panel.loc[panel["month"] == end_m, ["advertiser_id", "spend"]]
        merged = start_df.rename(columns={"spend": "spend_start"}).merge(
            end_df.rename(columns={"spend": "spend_end"}), on="advertiser_id"
        )
        merged["quarter"] = q
        merged["tenure_months"] = np.clip(start_m - merged["start_month"], 0, None)
        rows.append(merged)
    out = pd.concat(rows, ignore_index=True)
    both_active = (out["spend_start"] > 0) & (out["spend_end"] > 0)
    out["is_expander"] = (both_active & (out["spend_end"] > out["spend_start"])).astype(int)
    out["expansion_dollars"] = np.where(
        both_active & (out["spend_end"] > out["spend_start"]), out["spend_end"] - out["spend_start"], 0.0
    )
    out["is_large"] = (out["size_tier"] == "Large").astype(int)
    return out


def _add_trailing_growth(qf: pd.DataFrame) -> pd.DataFrame:
    """trailing_growth_rate for quarter q's features is that advertiser's
    own growth rate during quarter q-1 (the quarter being predicted FROM),
    never the quarter being predicted."""
    qf = qf.sort_values(["advertiser_id", "quarter"]).copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        qf["own_growth_rate"] = np.where(
            qf["spend_start"] > 0, (qf["spend_end"] - qf["spend_start"]) / qf["spend_start"], 0.0
        )
    qf["trailing_growth_rate"] = qf.groupby("advertiser_id")["own_growth_rate"].shift(1).fillna(0.0)
    qf["trailing_growth_rate"] = qf["trailing_growth_rate"].clip(-0.9, 3.0)
    return qf


def build_quarters(panel: pd.DataFrame) -> pd.DataFrame:
    return _add_trailing_growth(_quarter_frame(panel))


def train_propensity_model(train_quarters: pd.DataFrame) -> LogisticRegression:
    eligible = train_quarters.loc[train_quarters["spend_start"] > 0]
    X = eligible[FEATURES].to_numpy(dtype=float)
    y = eligible["is_expander"].to_numpy(dtype=int)
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model


def score(quarters: pd.DataFrame, model: LogisticRegression) -> np.ndarray:
    return model.predict_proba(quarters[FEATURES].to_numpy(dtype=float))[:, 1]


def top_decile_expansion_share(quarters_with_scores: pd.DataFrame) -> float:
    """Fraction of that quarter's actual expansion dollars held by the
    advertisers in the top 10% by predicted propensity score, among
    advertisers eligible that quarter (spend_start > 0)."""
    eligible = quarters_with_scores.loc[quarters_with_scores["spend_start"] > 0].copy()
    threshold = eligible["propensity_score"].quantile(0.90)
    top_decile = eligible.loc[eligible["propensity_score"] >= threshold]
    total_expansion = eligible["expansion_dollars"].sum()
    if total_expansion <= 0:
        return float("nan")
    return float(top_decile["expansion_dollars"].sum() / total_expansion)
