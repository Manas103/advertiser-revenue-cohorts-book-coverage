from revenue_cohorts.metrics import gini


def test_gini_zero_for_perfect_equality():
    assert abs(gini([10.0, 10.0, 10.0, 10.0])) < 1e-9


def test_gini_matches_hand_computed_value():
    # [0, 0, 0, 4]: classic textbook case, Gini = (n-1)/n = 0.75 for n=4
    # when all value sits on one unit (verified against the standard
    # closed-form for this exact degenerate distribution).
    assert abs(gini([0.0, 0.0, 0.0, 4.0]) - 0.75) < 1e-9


def test_gini_is_zero_for_all_zero_input():
    assert gini([0.0, 0.0, 0.0]) == 0.0


def test_gini_increases_with_more_inequality():
    low = gini([9.0, 10.0, 11.0])
    high = gini([1.0, 1.0, 28.0])
    assert high > low
