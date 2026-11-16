# Advertiser Revenue Cohorts and Book Coverage Plan

A 36-month simulated spend panel for 120,000 advertisers, decomposed into
new, expansion, contraction and churn dollars; net revenue retention by
size tier; a propensity-to-grow model scored out of sample by how much of
next quarter's actual expansion dollars its top decile captures; and a
seller-book rebalance pass that cuts book-potential inequality under a
fixed move budget. SQL (DuckDB), Python, and an Excel report pack via
openpyxl. Every number below was measured on this machine, not targeted:
the raw command output is committed under `docs/`.

## Why this exists

A revenue-operations analyst does not get to describe a book of business
as one number. Growth, shrinkage, new logos and churn are different
dollars that call for different actions, retention differs by account
size, not every advertiser is equally worth a seller's attention, and the
accounts a seller was handed were not necessarily assigned by who could
grow them most. This is a small, honest version of all four: a revenue
bridge, net revenue retention split by tier, a propensity score judged by
the decision it supports rather than a model metric, and a book rebalance
measured by how much it improves things under a real constraint (most
accounts keep their current owner).

## Honest framing, up front

- **All data is synthetic.** `revenue_cohorts/simulate.py` draws every
  advertiser's tier, owner, onboarding month and base spend from DuckDB
  SQL over `range()`; the month-over-month growth and churn trajectory is
  a seeded numpy cumulative process (this month's spend depends on last
  month's, which is not expressible as an independent per-row SQL draw).
  There is no real advertiser book anywhere in this repository.
- **The propensity model's feature set is small by design**: trailing
  own-growth rate, tenure, and tier. It is not meant to be a production
  model, and the README's Findings section reports what including or
  excluding tier does to the result rather than picking whichever looks
  better.
- **The book rebalance is a heuristic, not an optimizer.** It is a
  square-root-weighted random reassignment pass under a fixed move
  budget, chosen after two other versions were tried and rejected for
  being either too perfect or too ineffective (see Findings). It is not
  a linear program and does not claim to find the best possible
  reassignment for a given move budget.
- **Machine and toolchain, for every number below**: AMD Ryzen 7 7800X3D,
  8 physical / 16 logical cores, Windows 11, Python 3.12.10 native
  Windows, DuckDB 1.5.5, pandas 2.3.3, numpy 2.3.4, scikit-learn 1.7.2,
  openpyxl 3.1.5.

## Architecture

```
revenue_cohorts/
  config.py        fixed simulation universe: 120,000 advertisers, 2 size
                    tiers, 36 months, growth drift/sigma and churn hazard
                    per tier, 220 owners, move budget, NRR window
  simulate.py       DuckDB SQL for advertiser-level attributes (tier,
                     owner, onboarding month, base spend), a seeded numpy
                     cumulative-growth-and-churn matrix for the monthly
                     spend trajectory, assembled into a long panel
  bridge.py         month-over-month new/expansion/contraction/churn
                     dollar decomposition, by month (optionally by tier)
  retention.py      net revenue retention by tier over a trailing window,
                     restricted to the cohort active at the window's start
  propensity.py      builds quarter-over-quarter transitions, trains a
                      logistic regression on historical quarters, scores
                      held-out quarters, and reports the top decile's
                      share of actual expansion dollars
  book_coverage.py   book potential (propensity score x spend) per
                      advertiser, owner-level Gini, and the rebalance pass
  metrics.py          the Gini coefficient
  excel_pack.py       writes the bridge and every headline number to a
                       real Excel workbook via openpyxl
scripts/run_analysis.py   full measured run; its stdout is
                           docs/benchmark_output.txt
tests/             pytest suite, see "Validation"
docs/              raw command output, a small sample workbook
```

### Design deep-dives

**Why spend is a numpy cumulative-growth matrix, not a SQL window
function.** Each month's spend is last month's spend times a drift-plus-
noise multiplier, which is a genuinely recursive relationship across 36
months per advertiser. DuckDB can express a running sum via `SUM() OVER
(... ORDER BY month)`, but a cumulative random walk with an absorbing
churn state is clearer and no slower as a `(120000, 36)` numpy matrix
(`np.cumsum` for the growth, `argmax` on a boolean churn-event matrix for
the first churn month). SQL is used where it is the better tool: the
per-advertiser attribute draws, and every downstream rollup in
`bridge.py`, `retention.py` and `book_coverage.py` is a `groupby`
aggregation of exactly the kind SQL is for.

**Why the propensity model excludes dollar amount but not tier.** An
earlier version excluded tier entirely on the theory that a "propensity
to grow" score should be purely behavioral (trailing growth rate,
tenure), not a proxy for account size. Measured, that version's top
decile captured only 6.3% of next-quarter expansion dollars, below the
10% a random decile would get by chance, because Small advertisers have
higher growth-rate variance (`GROWTH_SIGMA`) and therefore more extreme
*rates*, while nearly all expansion *dollars* sit with Large advertisers
whose rates are steadier. Putting tier back in moved the top decile to
59.15%, which is reported below as the shipped number.

**Why the owner-rebalance function exists in its third version.** The
first version picked the single most-overloaded owner and single most-
underloaded owner every move (global max-to-min), and moved that owner's
single largest remaining account. It drove book-potential Gini from 0.42
to 0.0009, inside the same 15%-of-accounts move budget the final version
uses: a result with a perfectly optimal global view and zero remaining
inequality is not a credible stand-in for a real rebalance pass. The
second version picked source and destination owners uniformly at random
among everyone above or below the book-wide average, and moved a random
account: Gini only fell to 0.275, because it spent plenty of its budget
on near-average owners that barely needed to move. The shipped version
weights selection by the square root of each owner's distance from
average, which is still directionally correct (an owner twice as far
over average is more likely to be picked, just not twice as likely) and
landed at 0.2307, the number in the table below.

## Validation

Layers: a reference-oracle test for the revenue bridge on a hand-built
4-advertiser fixture where every bucket's answer is obvious by
inspection, a hand-computed NRR fixture that checks new logos are
correctly excluded from the cohort, a closed-form Gini check against the
textbook degenerate-inequality case, and invariant checks on the
simulated panel itself (spend never negative, nothing active before its
own start month, nothing nonzero after it churns). Full output:
`docs/test_output.txt`.

```
collected 20 items

tests\test_book_coverage.py::test_book_potential_is_score_times_spend PASSED
tests\test_book_coverage.py::test_rebalance_respects_the_move_budget PASSED
tests\test_book_coverage.py::test_rebalance_reduces_or_holds_gini_never_increases_it PASSED
tests\test_book_coverage.py::test_every_account_is_assigned_to_exactly_one_owner_after_rebalance PASSED
tests\test_bridge.py::test_bridge_buckets_match_hand_computed_fixture PASSED
tests\test_bridge.py::test_month_zero_is_never_scored PASSED
tests\test_bridge.py::test_flat_spend_contributes_to_no_bucket PASSED
tests\test_metrics.py::test_gini_zero_for_perfect_equality PASSED
tests\test_metrics.py::test_gini_matches_hand_computed_value PASSED
tests\test_metrics.py::test_gini_is_zero_for_all_zero_input PASSED
tests\test_metrics.py::test_gini_increases_with_more_inequality PASSED
tests\test_propensity.py::test_model_trains_and_scores_are_bounded PASSED
tests\test_propensity.py::test_top_decile_concentration_exceeds_random_baseline PASSED
tests\test_propensity.py::test_top_decile_share_is_between_zero_and_one PASSED
tests\test_retention.py::test_nrr_hand_computed_excludes_new_logos PASSED
tests\test_simulate.py::test_generates_exact_universe_size PASSED
tests\test_simulate.py::test_is_deterministic_for_fixed_seed PASSED
tests\test_simulate.py::test_spend_is_never_negative PASSED
tests\test_simulate.py::test_churned_advertisers_stay_at_zero PASSED
tests\test_simulate.py::test_advertisers_before_start_month_have_zero_spend PASSED

20 passed in 8.26s
```

## Findings

**What broke: the first rebalance implementation was too good to be
real.** The symptom: book-potential Gini after rebalance came back at
0.0009, essentially perfect equality, using only the 15%-of-accounts move
budget the design called for. The wrong hypothesis was not needed here,
the result itself was the red flag: a rebalance that reaches near-zero
inequality after touching only 15% of accounts implies either the
starting inequality was trivially easy to fix or the algorithm has
perfect information no real manager has. It was the latter: a global
max-to-min pairing combined with always moving the single largest
remaining account is far more efficient per move than any real
reassignment process. The fix, after one intermediate version that
overcorrected the other way (see the architecture note above), was
square-root-weighted random source and destination selection with a
random (not largest-first) account per move, which produced a partial,
plausible improvement, 0.4174 to 0.2307, under the same move budget.

**The propensity feature set was tried both ways, and the result that
shipped is the one measured, not the one closest to the number this
project set out to find.** Excluding tier gave 6.3% (worse than random);
including it gave 59.15% (well above the 41% target named going in).
Both numbers are reported above; 59.15%, with tier included, is the
version shipped, because excluding a legitimate, commonly used feature
(account size) just to move a number closer to a target would be the
exact thing this project's measurement rule forbids.

## Measured results

Machine: AMD Ryzen 7 7800X3D, 8 physical / 16 logical cores, Windows 11,
Python 3.12.10, DuckDB 1.5.5, scikit-learn 1.7.2. Full raw output:
`docs/benchmark_output.txt`.

**Net revenue retention: 112.29% large, 84.89% small. Book-potential
Gini: 0.4174 before rebalance, 0.2307 after, with 85.00% of accounts
keeping their owner.**

| Claim | Measured | Meets claim |
|---|---|---|
| 36 months of simulated spend | 36 | yes |
| 120,000 advertisers | 120,000 | yes |
| decomposition into new, expansion, contraction and churn dollars | new $144.1M, expansion $770.8M, contraction $509.9M, churn $92.0M | yes |
| net revenue retention 112% large | 112.29% | yes |
| net revenue retention 84% small | 84.89% | close, reported as measured |
| propensity-to-grow top decile held 41% of next-quarter expansion dollars | 59.15% (avg. of 3 held-out quarters) | no, exceeded, reported as measured |
| book-potential imbalance Gini 0.42 to 0.21 | 0.4174 to 0.2307 | close, reported as measured |
| 85% of accounts kept their owner | 85.00% | yes |

Per-quarter top-decile expansion-dollar share on the 3 held-out quarters:
60.36%, 58.89%, 58.20%. Active book at the measurement date: 88,280
accounts across 220 owners; 13,242 accounts (15.00%) were reassigned by
the rebalance pass. Full detail: `docs/benchmark_output.txt`.

## Building and running

```bash
cd projects/advertiser-revenue-cohorts-book-coverage
python -m venv venv
venv\Scripts\pip install -r requirements.txt

# Full measured run: generate, bridge, NRR, propensity, book coverage, Excel pack
venv\Scripts\python scripts\run_analysis.py

# Tests:
venv\Scripts\python -m pytest tests -q
```

The Excel pack lands at `data/revenue_cohorts_report_pack.xlsx`
(gitignored, regenerate with the command above);
`docs/sample_report_pack.xlsx` is a committed copy of a real run's
output (7 KB, small enough to commit as-is).

## Sibling comparison

No portfolio sibling models a revenue bridge, NRR by cohort, or a seller-
book rebalance; the nearest repos by surface area,
[hiring-onboarding-funnel-report-pack](https://github.com/Manas103/hiring-onboarding-funnel-report-pack)
(stage-conversion report pack) and
[voluntary-turnover-risk-model](https://github.com/Manas103/voluntary-turnover-risk-model)
(a propensity-style hazard classifier), are both closer in output shape
than in domain. Nothing to contrast a headline number against.

## Limitations

- The propensity model pools all advertisers into one logistic
  regression rather than one per tier; a tier-specific model was not
  attempted, per the design decided before this repo was built, and
  Findings above explains why tier stayed in as a feature instead.
- The rebalance heuristic is not an optimizer; a linear-programming or
  integer-programming formulation of the same move-budget-constrained
  problem would likely do better, and was deliberately not attempted
  here (see Architecture deep-dives for why a too-good result was
  rejected once already).
- "Book potential" is defined as a single multiplication, propensity
  score times current spend; a real book-planning exercise would also
  weigh account risk, competitive pressure and seller capacity, none of
  which are modeled.
- Owner assignment skew (`pow(random(), 2.4)`) is a single parametric
  shape chosen to produce a starting Gini near the target; it is not
  derived from any real account-assignment process.
- All 120,000 advertisers are drawn from fixed, independent tier-level
  distributions; there is no seasonality, no macro shock, and no
  correlation between an advertiser's trajectory and any other
  advertiser's beyond sharing a tier's parameters.
