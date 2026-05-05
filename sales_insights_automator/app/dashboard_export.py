"""
CSV, ZIP, and PDF exports for the Analysis Dashboard.

All builders are pure functions (no Streamlit imports) so they stay easy to test.
"""

from __future__ import annotations

import io
import re
import zipfile
from datetime import datetime, timezone
from typing import Any, Mapping, Optional

import pandas as pd
from fpdf import FPDF, XPos, YPos

from app.components.charts import _label

# ── Filenames & KPI display names ─────────────────────────────────────────────

_KPI_LABELS: dict[str, str] = {
    "total_revenue": "Total Revenue ($)",
    "total_orders": "Total Orders",
    "average_order_value": "Average Order Value ($)",
    "total_units_sold": "Units Sold",
    "average_discount_pct": "Average Discount (%)",
    "average_unit_price": "Average Unit Price ($)",
    "median_order_value": "Median Order Value ($)",
    "min_order_value": "Min Order Value ($)",
    "max_order_value": "Max Order Value ($)",
}


def safe_export_basename(file_name: str) -> str:
    """Strip path/extension and keep a filesystem-safe stem."""
    stem = file_name.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    stem = re.sub(r"[^\w\-]+", "_", stem).strip("_")
    return (stem[:80] or "sales_dashboard")


def dataframe_with_display_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Copy with columns renamed using chart display labels where defined."""
    if df is None or df.empty:
        return df.copy() if df is not None else pd.DataFrame()
    renamer = {c: _label(c) for c in df.columns}
    return df.rename(columns=renamer)


def filtered_transactions_csv_bytes(df: pd.DataFrame) -> bytes:
    """UTF-8 CSV of the filtered transaction-level data (human-readable headers)."""
    out = dataframe_with_display_columns(df)
    return out.to_csv(index=False).encode("utf-8")


def kpi_summary_csv_bytes(live_stats: Mapping[str, Any]) -> bytes:
    """Single-row CSV of headline KPIs with readable column names."""
    row: dict[str, Any] = {}
    for k, v in live_stats.items():
        label = _KPI_LABELS.get(k, k.replace("_", " ").title())
        row[label] = v
    return pd.DataFrame([row]).to_csv(index=False).encode("utf-8")


def _df_to_csv_bytes(df: Optional[pd.DataFrame]) -> Optional[bytes]:
    if df is None or df.empty:
        return None
    disp = dataframe_with_display_columns(df)
    return disp.to_csv(index=False).encode("utf-8")


def _safe_zip_member(name: str) -> str:
    return re.sub(r"[^\w\-.]+", "_", name)[:200]


def dashboard_zip_bytes(
    *,
    live_stats: Mapping[str, Any],
    fdf: pd.DataFrame,
    dims: dict[str, tuple[str, Optional[pd.DataFrame]]],
    monthly: Optional[pd.DataFrame],
    quarterly: Optional[pd.DataFrame],
    dow: Optional[pd.DataFrame],
    rep_perf: Optional[pd.DataFrame],
    crosstab: Optional[pd.DataFrame],
    filter_note: str,
    base_name: str,
) -> bytes:
    """ZIP containing KPIs, transactions, and tabular aggregates matching the dashboard."""
    buf = io.BytesIO()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    readme = (
        f"Sales Insights — dashboard export\n"
        f"Source dataset: {base_name}\n"
        f"Generated: {stamp}\n"
        f"{filter_note}\n"
        f"Rows in filtered slice: {len(fdf):,}\n"
    )
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("README.txt", readme.encode("utf-8"))
        zf.writestr("kpis.csv", kpi_summary_csv_bytes(live_stats))
        if fdf is not None and not fdf.empty:
            zf.writestr("transactions.csv", filtered_transactions_csv_bytes(fdf))

        mcsv = _df_to_csv_bytes(monthly)
        if mcsv:
            zf.writestr("monthly_revenue.csv", mcsv)
        qcsv = _df_to_csv_bytes(quarterly)
        if qcsv:
            zf.writestr("quarterly_revenue.csv", qcsv)
        dcsv = _df_to_csv_bytes(dow)
        if dcsv:
            zf.writestr("revenue_by_day_of_week.csv", dcsv)
        rcsv = _df_to_csv_bytes(rep_perf)
        if rcsv:
            zf.writestr("salesperson_performance.csv", rcsv)
        xcsv = _df_to_csv_bytes(crosstab)
        if xcsv:
            zf.writestr("category_x_region_revenue.csv", xcsv)

        for tab_label, (_col, data) in dims.items():
            b = _df_to_csv_bytes(data)
            if b:
                fname = _safe_zip_member(f"revenue_by_{tab_label}") + ".csv"
                zf.writestr(fname, b)

    return buf.getvalue()


def _sanitize_pdf_text(s: str) -> str:
    """Keep PDF text compatible with core fonts (Latin-1)."""
    if not s:
        return ""
    return s.encode("latin-1", errors="replace").decode("latin-1")


class _DashPDF(FPDF):
    def footer(self) -> None:
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(
            0,
            10,
            f"Page {self.page_no()}/{{nb}}",
            align="C",
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
        )


def _pdf_table(pdf: FPDF, headers: list[str], rows: list[list[str]], col_widths: list[float]) -> None:
    pdf.set_font("Helvetica", "B", 9)
    for i, h in enumerate(headers):
        pdf.cell(col_widths[i], 7, _sanitize_pdf_text(h), border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    for row in rows:
        for i, cell in enumerate(row):
            pdf.cell(col_widths[i], 6, _sanitize_pdf_text(str(cell)), border=1)
        pdf.ln()


def dashboard_pdf_bytes(
    *,
    live_stats: Mapping[str, Any],
    dims: dict[str, tuple[str, Optional[pd.DataFrame]]],
    monthly: Optional[pd.DataFrame],
    quarterly: Optional[pd.DataFrame],
    dow: Optional[pd.DataFrame],
    filter_note: str,
    base_name: str,
    max_rows_per_table: int = 15,
) -> bytes:
    """Multi-page PDF: KPIs + compact tables (charts are not rendered)."""
    pdf = _DashPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=14)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(
        0,
        10,
        _sanitize_pdf_text("Sales Insights Dashboard"),
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(
        0,
        6,
        _sanitize_pdf_text(f"Source: {base_name}"),
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )
    pdf.multi_cell(0, 6, _sanitize_pdf_text(filter_note))
    pdf.ln(2)

    # KPI block
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(
        0,
        8,
        "Key metrics (current view)",
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )
    pdf.set_font("Helvetica", "", 10)
    kpi_order = [
        "total_revenue",
        "total_orders",
        "average_order_value",
        "total_units_sold",
        "average_discount_pct",
        "median_order_value",
    ]
    for key in kpi_order:
        if key not in live_stats:
            continue
        label = _KPI_LABELS.get(key, key)
        val = live_stats[key]
        if isinstance(val, float):
            val_s = f"{val:,.2f}"
        else:
            val_s = f"{val:,}" if isinstance(val, int) else str(val)
        pdf.cell(
            0,
            6,
            f"  {label}: {val_s}",
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
        )
    pdf.ln(4)

    def _emit_df_section(title: str, df: Optional[pd.DataFrame]) -> None:
        if df is None or df.empty:
            return
        disp = dataframe_with_display_columns(df.head(max_rows_per_table))
        if disp.empty:
            return
        if pdf.get_y() > 250:
            pdf.add_page()
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(
            0,
            7,
            _sanitize_pdf_text(title),
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
        )
        cols = list(disp.columns)
        w_avail = 190
        cw = [max(28, w_avail / len(cols))] * len(cols)
        s = cw[0] * len(cols)
        if s > w_avail:
            scale = w_avail / s
            cw = [c * scale for c in cw]
        rows = disp.astype(str).values.tolist()
        _pdf_table(pdf, [str(c) for c in cols], rows, cw)
        pdf.ln(3)
        if len(df) > max_rows_per_table:
            pdf.set_font("Helvetica", "I", 8)
            pdf.cell(
                0,
                5,
                f"(Showing first {max_rows_per_table} of {len(df)} rows — see ZIP/CSV for full data.)",
                new_x=XPos.LMARGIN,
                new_y=YPos.NEXT,
            )
            pdf.set_font("Helvetica", "", 10)
            pdf.ln(2)

    _emit_df_section("Monthly revenue (sample)", monthly)
    _emit_df_section("Quarterly revenue (sample)", quarterly)
    _emit_df_section("Revenue by day of week", dow)

    for tab_label, (_col, data) in dims.items():
        _emit_df_section(f"Revenue by {tab_label} (sample)", data)

    pdf.set_font("Helvetica", "I", 9)
    pdf.ln(4)
    pdf.multi_cell(
        0,
        5,
        _sanitize_pdf_text(
            "Note: This PDF contains tables only. Download the ZIP or CSV "
            "for complete data and use the app for interactive charts."
        ),
    )

    raw = pdf.output()
    if isinstance(raw, (bytes, bytearray)):
        return bytes(raw)
    return str(raw).encode("latin-1")

