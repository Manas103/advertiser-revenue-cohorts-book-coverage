"""Fixed simulation constants shared across the package.

Changing any of these changes the simulated world; they live in one place
so the generator, the models and the README's stated parameters cannot
drift apart.
"""

RANDOM_SEED = 42

N_ADVERTISERS = 120_000
N_MONTHS = 36

SIZE_TIERS = ["Large", "Small"]
SIZE_TIER_WEIGHTS = {"Large": 0.12, "Small": 0.88}

# Monthly base spend, lognormal(mu, sigma) in dollars.
BASE_SPEND_MU = {"Large": 10.5, "Small": 7.2}   # exp(10.5) ~ $36K/mo, exp(7.2) ~ $1.3K/mo
BASE_SPEND_SIGMA = {"Large": 0.55, "Small": 0.65}

# Monthly multiplicative drift and noise on log-spend while active and not
# churned. Large advertisers grow slowly on average; Small advertisers
# shrink slowly on average, which is what drives the NRR gap between tiers.
GROWTH_DRIFT = {"Large": 0.0105, "Small": -0.0062}
GROWTH_SIGMA = {"Large": 0.05, "Small": 0.09}

# Monthly hazard of permanent churn (spend goes to 0 and stays there),
# checked only after MIN_TENURE_MONTHS.
CHURN_HAZARD = {"Large": 0.0020, "Small": 0.0115}
MIN_TENURE_MONTHS = 3

# Most advertisers are present from month 0 (an existing book); the rest
# start in a later month, uniformly, giving "new" dollars throughout the
# panel instead of only in month 0.
NEW_LOGO_SHARE = 0.18
LATEST_START_MONTH = 30

N_OWNERS = 220

# Book-potential imbalance: the fraction of accounts a rebalance pass is
# allowed to move to a different owner.
MAX_REASSIGNMENT_SHARE = 0.15

# NRR measurement window: trailing TRAILING_MONTHS vs the TRAILING_MONTHS
# immediately before that, restricted to the cohort active at the start of
# the earlier window (excludes new logos, per the standard NRR definition).
TRAILING_MONTHS = 12
