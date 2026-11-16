import pandas as pd

from revenue_cohorts.retention import compute_nrr


def test_nrr_hand_computed_excludes_new_logos():
    # month 0 (start) to month 12 (end). Advertiser 3 starts mid-window
    # (spend_start == 0) and must be excluded from the cohort.
    rows = []
    for m in [0, 12]:
        rows += [
            {"advertiser_id": 1, "size_tier": "Large", "spend": 100.0 if m == 0 else 120.0, "month": m},
            {"advertiser_id": 2, "size_tier": "Large", "spend": 100.0 if m == 0 else 80.0, "month": m},
            {"advertiser_id": 3, "size_tier": "Large", "spend": 0.0 if m == 0 else 500.0, "month": m},
        ]
    panel = pd.DataFrame(rows)
    nrr = compute_nrr(panel, trailing_months=12)
    # cohort = advertisers 1 and 2 only. start=200, end=200 -> 100%.
    assert abs(nrr["Large"] - 100.0) < 1e-9
