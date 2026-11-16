"""Reference-oracle test: a tiny hand-built panel where the new / expansion
/ contraction / churn split is obvious by inspection, diffed exactly
against monthly_bridge's output."""
import pandas as pd

from revenue_cohorts.bridge import monthly_bridge


def test_bridge_buckets_match_hand_computed_fixture():
    panel = pd.DataFrame({
        "advertiser_id": [1, 1, 2, 2, 3, 3, 4, 4],
        "month": [0, 1, 0, 1, 0, 1, 0, 1],
        "spend": [
            100.0, 150.0,  # advertiser 1: expansion of 50
            200.0, 120.0,  # advertiser 2: contraction of 80
            0.0, 60.0,     # advertiser 3: new business of 60
            90.0, 0.0,     # advertiser 4: churn of 90
        ],
    })
    out = monthly_bridge(panel)
    row = out.loc[out["month"] == 1].iloc[0]
    assert abs(row["expansion_dollars"] - 50.0) < 1e-9
    assert abs(row["contraction_dollars"] - 80.0) < 1e-9
    assert abs(row["new_dollars"] - 60.0) < 1e-9
    assert abs(row["churn_dollars"] - 90.0) < 1e-9


def test_month_zero_is_never_scored():
    panel = pd.DataFrame({
        "advertiser_id": [1, 1],
        "month": [0, 1],
        "spend": [100.0, 100.0],
    })
    out = monthly_bridge(panel)
    assert 0 not in out["month"].to_numpy()


def test_flat_spend_contributes_to_no_bucket():
    panel = pd.DataFrame({
        "advertiser_id": [1, 1],
        "month": [0, 1],
        "spend": [100.0, 100.0],
    })
    out = monthly_bridge(panel)
    row = out.iloc[0]
    assert row["new_dollars"] == 0.0
    assert row["expansion_dollars"] == 0.0
    assert row["contraction_dollars"] == 0.0
    assert row["churn_dollars"] == 0.0
