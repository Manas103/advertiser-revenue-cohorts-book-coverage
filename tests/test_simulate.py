from revenue_cohorts import config
from revenue_cohorts.simulate import generate_panel


def test_generates_exact_universe_size():
    panel = generate_panel()
    assert panel["advertiser_id"].nunique() == config.N_ADVERTISERS
    assert panel["month"].nunique() == config.N_MONTHS
    assert len(panel) == config.N_ADVERTISERS * config.N_MONTHS


def test_is_deterministic_for_fixed_seed():
    a = generate_panel()
    b = generate_panel()
    assert a["spend"].to_numpy().tolist() == b["spend"].to_numpy().tolist()


def test_spend_is_never_negative():
    panel = generate_panel()
    assert (panel["spend"] >= 0).all()


def test_churned_advertisers_stay_at_zero():
    panel = generate_panel().sort_values(["advertiser_id", "month"])
    zero_then = panel.groupby("advertiser_id")["spend"].apply(lambda s: (s == 0).to_numpy())
    for arr in zero_then:
        if arr.any():
            first_zero = arr.argmax()
            assert arr[first_zero:].all() or not arr[:first_zero].any()


def test_advertisers_before_start_month_have_zero_spend():
    panel = generate_panel()
    before_start = panel.loc[panel["month"] < panel["start_month"]]
    assert (before_start["spend"] == 0).all()
