import pandas as pd

from revenue_cohorts.book_coverage import book_potential, owner_totals, rebalance
from revenue_cohorts.metrics import gini


def test_book_potential_is_score_times_spend():
    df = pd.DataFrame({"advertiser_id": [1], "propensity_score": [0.5], "spend": [1000.0]})
    out = book_potential(df)
    assert out["book_potential"].iloc[0] == 500.0


def test_rebalance_respects_the_move_budget():
    accounts = pd.DataFrame({
        "advertiser_id": range(100),
        "owner_id": [0] * 80 + [1] * 20,
        "book_potential": [100.0] * 80 + [10.0] * 20,
    })
    new_owner, n_moved = rebalance(accounts, max_reassignment_share=0.10)
    assert n_moved <= 10


def test_rebalance_reduces_or_holds_gini_never_increases_it():
    accounts = pd.DataFrame({
        "advertiser_id": range(200),
        "owner_id": [0] * 150 + [1] * 50,
        "book_potential": [100.0] * 150 + [10.0] * 50,
    })
    before = gini(owner_totals(accounts).to_numpy())
    new_owner, n_moved = rebalance(accounts, max_reassignment_share=0.20)
    accounts2 = accounts.copy()
    accounts2["owner_id"] = accounts2["advertiser_id"].map(new_owner)
    after = gini(owner_totals(accounts2).to_numpy())
    assert after <= before


def test_every_account_is_assigned_to_exactly_one_owner_after_rebalance():
    accounts = pd.DataFrame({
        "advertiser_id": range(50),
        "owner_id": [0] * 30 + [1] * 20,
        "book_potential": [5.0] * 50,
    })
    new_owner, _ = rebalance(accounts, max_reassignment_share=0.15)
    assert set(new_owner.index) == set(accounts["advertiser_id"])
