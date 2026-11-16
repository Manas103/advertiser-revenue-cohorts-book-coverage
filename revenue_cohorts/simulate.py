"""Generates the 120,000-advertiser x 36-month simulated spend panel.

Advertiser-level attributes (tier, owner, onboarding month, base spend) are
drawn as DuckDB SQL over range(), vectorized, never a Python row-by-row
loop. The month-over-month growth and churn trajectory is a cumulative
process by construction (this month's spend depends on last month's), so
it is built as a (N_ADVERTISERS, N_MONTHS) numpy matrix instead of forcing
a recursive relationship through SQL window functions; the advertiser-month
long table that every downstream module reads is assembled from that
matrix. All random draws come from one seeded numpy Generator so a fixed
RANDOM_SEED reproduces the exact panel.
"""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from . import config


def _weighted_case_sql(rand_col: str, weights: dict[str, float]) -> str:
    items = list(weights.items())
    clauses = []
    cum = 0.0
    for key, w in items[:-1]:
        cum += w
        clauses.append(f"WHEN {rand_col} < {cum} THEN '{key}'")
    fallback = items[-1][0]
    return "CASE " + " ".join(clauses) + f" ELSE '{fallback}' END"


def _generate_advertisers() -> pd.DataFrame:
    con = duckdb.connect(":memory:")
    con.execute(f"SELECT setseed({config.RANDOM_SEED / 2147483647.0})")
    tier_case = _weighted_case_sql("r_tier", config.SIZE_TIER_WEIGHTS)
    df = con.execute(f"""
        SELECT
            i AS advertiser_id,
            {tier_case} AS size_tier,
            -- Account assignment history is skewed, not uniform: a book
            -- built up by seniority and tenure, not by growth potential,
            -- which is exactly what creates the book-potential imbalance
            -- book_coverage.py measures and then rebalances.
            CAST(floor({config.N_OWNERS} * pow(random(), 2.4)) AS INTEGER) AS owner_id,
            CASE WHEN r_new < {config.NEW_LOGO_SHARE}
                 THEN CAST(1 + floor(random() * {config.LATEST_START_MONTH}) AS INTEGER)
                 ELSE 0 END AS start_month,
            r_base_1, r_base_2
        FROM (
            SELECT i, random() AS r_tier, random() AS r_new, random() AS r_base_1, random() AS r_base_2
            FROM range({config.N_ADVERTISERS}) t(i)
        )
    """).fetchdf()
    con.close()

    mu = df["size_tier"].map(config.BASE_SPEND_MU)
    sigma = df["size_tier"].map(config.BASE_SPEND_SIGMA)
    z = np.sqrt(-2.0 * np.log(df["r_base_1"].clip(1e-9, 1 - 1e-9))) * np.cos(2 * np.pi * df["r_base_2"])
    df["base_spend_log"] = mu + sigma * z
    return df.drop(columns=["r_base_1", "r_base_2"])


def generate_panel() -> pd.DataFrame:
    """Returns a long DataFrame: one row per (advertiser_id, month), with
    size_tier, owner_id and spend (0.0 once churned or before start_month).
    """
    adv = _generate_advertisers()
    n = len(adv)
    t = config.N_MONTHS
    rng = np.random.default_rng(config.RANDOM_SEED + 7)

    drift = adv["size_tier"].map(config.GROWTH_DRIFT).to_numpy()[:, None]
    sigma = adv["size_tier"].map(config.GROWTH_SIGMA).to_numpy()[:, None]
    noise = rng.standard_normal((n, t))
    monthly_growth = drift + sigma * noise  # (n, t) log-growth applied at each month index

    start_month = adv["start_month"].to_numpy()
    months = np.arange(t)[None, :]
    active = months >= start_month[:, None]

    # tenure_index: months since start_month, used to zero out growth before
    # an advertiser has started and to accumulate growth only from then on.
    tenure_index = np.clip(months - start_month[:, None], 0, None)
    growth_since_start = np.where(active, monthly_growth, 0.0)
    cum_growth = np.cumsum(growth_since_start, axis=1)

    churn_hazard = adv["size_tier"].map(config.CHURN_HAZARD).to_numpy()[:, None]
    churn_eligible = active & (tenure_index >= config.MIN_TENURE_MONTHS)
    churn_draw = rng.random((n, t))
    churn_event = churn_eligible & (churn_draw < churn_hazard)
    # first churn month per advertiser (np.inf if it never churns).
    first_churn = np.where(churn_event.any(axis=1), churn_event.argmax(axis=1), t)
    churned = months >= first_churn[:, None]

    spend = np.where(active & ~churned, np.exp(adv["base_spend_log"].to_numpy()[:, None] + cum_growth), 0.0)
    spend = np.round(spend, 2)

    long = pd.DataFrame({
        "advertiser_id": np.repeat(adv["advertiser_id"].to_numpy(), t),
        "month": np.tile(months.ravel(), n),
        "spend": spend.ravel(),
    })
    long = long.merge(adv[["advertiser_id", "size_tier", "owner_id", "start_month"]], on="advertiser_id")
    return long
