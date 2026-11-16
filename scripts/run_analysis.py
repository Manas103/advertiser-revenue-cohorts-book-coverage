"""Full measured run: generate the 36-month panel, decompose the revenue
bridge, compute NRR by tier, train and score the propensity model, rebalance
seller books, and write the Excel report pack. Run with:

    python scripts/run_analysis.py

stdout is committed verbatim as docs/benchmark_output.txt.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from revenue_cohorts import config
from revenue_cohorts.book_coverage import book_potential, owner_totals, rebalance
from revenue_cohorts.bridge import monthly_bridge
from revenue_cohorts.excel_pack import write_report_pack
from revenue_cohorts.metrics import gini
from revenue_cohorts.propensity import build_quarters, score, top_decile_expansion_share, train_propensity_model
from revenue_cohorts.retention import compute_nrr
from revenue_cohorts.simulate import generate_panel

TRAIN_QUARTERS = range(1, 9)
HOLDOUT_QUARTERS = range(9, 12)


def main() -> None:
    t0 = time.perf_counter()
    panel = generate_panel()
    gen_s = time.perf_counter() - t0
    print("=== Generation ===")
    print(f"{len(panel)} advertiser-month rows, {panel['advertiser_id'].nunique()} advertisers, "
          f"{config.N_MONTHS} months, generated in {gen_s:.2f}s")
    print(panel["size_tier"].value_counts().to_string())

    t0 = time.perf_counter()
    bridge = monthly_bridge(panel)
    bridge_s = time.perf_counter() - t0
    print(f"\n=== Revenue bridge ({bridge_s:.2f}s) ===")
    totals = bridge[["new_dollars", "expansion_dollars", "contraction_dollars", "churn_dollars"]].sum()
    print(totals.to_string())

    print("\n=== Net revenue retention (trailing 12 months) ===")
    nrr = compute_nrr(panel)
    for tier, value in nrr.items():
        print(f"{tier}: {value:.2f}%")

    t0 = time.perf_counter()
    qf = build_quarters(panel)
    print(f"\n=== Propensity model ({time.perf_counter() - t0:.2f}s to build {len(qf)} quarter rows) ===")
    train = qf.loc[qf["quarter"].isin(TRAIN_QUARTERS)]
    holdout = qf.loc[qf["quarter"].isin(HOLDOUT_QUARTERS)].copy()
    model = train_propensity_model(train)
    holdout["propensity_score"] = score(holdout, model)
    shares = [top_decile_expansion_share(holdout.loc[holdout["quarter"] == q]) for q in HOLDOUT_QUARTERS]
    for q, s in zip(HOLDOUT_QUARTERS, shares):
        print(f"quarter {q} (holdout): top-decile expansion-dollar share {s*100:.2f}%")
    avg_share = float(np.mean(shares))
    print(f"average holdout top-decile expansion-dollar share: {avg_share*100:.2f}%")

    print("\n=== Book coverage ===")
    latest = qf.loc[qf["quarter"] == qf["quarter"].max()].copy()
    latest["propensity_score"] = score(latest, model)
    latest["spend"] = latest["spend_end"]
    active = latest.loc[latest["spend"] > 0]
    bp = book_potential(active)
    before_totals = owner_totals(bp)
    gini_before = gini(before_totals.to_numpy())
    print(f"active accounts: {len(bp)}, owners: {len(before_totals)}")
    print(f"book-potential Gini before rebalance: {gini_before:.4f}")

    new_owner, n_moved = rebalance(bp)
    pct_kept = 1 - n_moved / len(bp)
    bp_after = bp.copy()
    bp_after["owner_id"] = bp_after["advertiser_id"].map(new_owner)
    gini_after = gini(owner_totals(bp_after).to_numpy())
    print(f"accounts moved: {n_moved} of {len(bp)} ({pct_kept*100:.2f}% kept their owner)")
    print(f"book-potential Gini after rebalance: {gini_after:.4f}")

    out_path = Path(__file__).resolve().parents[1] / "data" / "revenue_cohorts_report_pack.xlsx"
    write_report_pack(out_path, bridge, nrr, avg_share, gini_before, gini_after, pct_kept)
    print(f"\nExcel pack written to {out_path} ({out_path.stat().st_size} bytes)")

    print("\n=== Claims summary ===")
    print(f"36 months of simulated spend: {config.N_MONTHS}")
    print(f"120,000 advertisers: {panel['advertiser_id'].nunique()}")
    print(f"decomposition into new/expansion/contraction/churn dollars: new ${totals['new_dollars']:,.2f}, "
          f"expansion ${totals['expansion_dollars']:,.2f}, contraction ${totals['contraction_dollars']:,.2f}, "
          f"churn ${totals['churn_dollars']:,.2f}")
    print(f"net revenue retention large: {nrr['Large']:.2f}%")
    print(f"net revenue retention small: {nrr['Small']:.2f}%")
    print(f"propensity top decile holds: {avg_share*100:.2f}% of next-quarter expansion dollars")
    print(f"book-potential imbalance Gini: {gini_before:.4f} to {gini_after:.4f}")
    print(f"accounts that kept their owner: {pct_kept*100:.2f}%")


if __name__ == "__main__":
    main()
