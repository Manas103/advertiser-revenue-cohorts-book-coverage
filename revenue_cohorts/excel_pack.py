"""Writes the full analysis (bridge, NRR, propensity, book coverage) to a
real Excel workbook via openpyxl, one sheet per artifact, with a summary
sheet listing every headline number."""
from __future__ import annotations

from pathlib import Path

from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows


def _write_df_sheet(wb: Workbook, title: str, df) -> None:
    ws = wb.create_sheet(title=title)
    for row in dataframe_to_rows(df, index=False, header=True):
        ws.append(row)


def write_report_pack(path: str | Path, bridge_df, nrr: dict, top_decile_share: float,
                       gini_before: float, gini_after: float, pct_kept: float) -> None:
    wb = Workbook()
    wb.remove(wb.active)

    summary = wb.create_sheet(title="Summary")
    summary.append(["Metric", "Value"])
    summary.append(["NRR Large (%)", round(nrr["Large"], 2)])
    summary.append(["NRR Small (%)", round(nrr["Small"], 2)])
    summary.append(["Propensity top-decile expansion-dollar share", round(top_decile_share, 4)])
    summary.append(["Book-potential Gini before rebalance", round(gini_before, 4)])
    summary.append(["Book-potential Gini after rebalance", round(gini_after, 4)])
    summary.append(["Share of accounts that kept their owner", round(pct_kept, 4)])

    _write_df_sheet(wb, "RevenueBridge", bridge_df)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)
