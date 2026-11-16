import numpy as np
import pandas as pd

from revenue_cohorts.propensity import FEATURES, score, top_decile_expansion_share, train_propensity_model


def _toy_quarters():
    rng = np.random.default_rng(0)
    n = 1000
    growth = rng.uniform(-0.3, 1.0, size=n)
    tenure = rng.integers(0, 24, size=n)
    is_large = rng.integers(0, 2, size=n)
    p = np.clip(0.1 + 0.5 * growth, 0.01, 0.95)
    is_expander = (rng.random(n) < p).astype(int)
    expansion_dollars = np.where(is_expander == 1, rng.uniform(100, 10_000, size=n), 0.0)
    return pd.DataFrame({
        "trailing_growth_rate": growth,
        "tenure_months": tenure,
        "is_large": is_large,
        "is_expander": is_expander,
        "expansion_dollars": expansion_dollars,
        "spend_start": 1000.0,
    })


def test_model_trains_and_scores_are_bounded():
    toy = _toy_quarters()
    model = train_propensity_model(toy)
    s = score(toy, model)
    assert ((s >= 0.0) & (s <= 1.0)).all()


def test_top_decile_concentration_exceeds_random_baseline():
    """On data engineered so growth drives both the label and the dollar
    amount, the top decile by score must capture well above a naive 10%
    of expansion dollars."""
    toy = _toy_quarters()
    model = train_propensity_model(toy)
    toy = toy.copy()
    toy["propensity_score"] = score(toy, model)
    share = top_decile_expansion_share(toy)
    assert share > 0.10


def test_top_decile_share_is_between_zero_and_one():
    toy = _toy_quarters()
    model = train_propensity_model(toy)
    toy["propensity_score"] = score(toy, model)
    share = top_decile_expansion_share(toy)
    assert 0.0 <= share <= 1.0
