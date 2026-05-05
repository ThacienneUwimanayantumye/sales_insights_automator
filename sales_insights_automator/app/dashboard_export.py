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
import plotly.graph_objects as go
from fpdf import FPDF, XPos, YPos

from app.components.charts import _label
from app.dashboard_pdf_figures import collect_dashboard_figures


class DashboardPdfError(RuntimeError):
    """PDF export failed (e.g. Kaleido not installed or chart render error)."""

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


def figure_to_png_bytes(
    fig: go.Figure,
    *,
    width: int = 1280,
    height: int = 720,
) -> bytes:
    """Rasterise a Plotly figure to PNG via Kaleido (required)."""
    try:
        out = fig.to_image(format="png", width=width, height=height)
    except Exception as e:
        raise DashboardPdfError(
            "Could not render charts for PDF. Install Kaleido: pip install kaleido"
        ) from e
    return bytes(out) if isinstance(out, (bytes, bytearray)) else out


def dashboard_pdf_bytes(
    *,
    live_stats: Mapping[str, Any],
    fdf: pd.DataFrame,
    result: Any,
    dims: dict[str, tuple[str, Optional[pd.DataFrame]]],
    monthly: Optional[pd.DataFrame],
    quarterly: Optional[pd.DataFrame],
    dow: Optional[pd.DataFrame],
    rep_perf: Optional[pd.DataFrame],
    crosstab: Optional[pd.DataFrame],
    extra_dims: list[str],
    extra_metrics: list[str],
    filter_note: str,
    base_name: str,
    chart_width: int = 1280,
    chart_height: int = 720,
) -> bytes:
    """Multi-page PDF: KPI summary plus every dashboard chart as an embedded image."""
    figures = collect_dashboard_figures(
        fdf=fdf,
        result=result,
        dims=dims,
        monthly=monthly,
        quarterly=quarterly,
        dow_chart=dow,
        rep_perf_df=rep_perf,
        crosstab_df=crosstab,
        extra_dims=list(extra_dims or []),
        extra_metrics=list(extra_metrics or []),
    )

    pdf = _DashPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=14)

    # ── Cover + KPIs ──────────────────────────────────────────────────────────
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
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(
        0,
        5,
        _sanitize_pdf_text(
            "Following pages: static copies of the dashboard charts (current filters). "
            "For raw data use CSV or ZIP export."
        ),
    )

    # ── One chart per page ─────────────────────────────────────────────────────
    img_w_mm = 190.0
    for section, subtitle, fig in figures:
        png = figure_to_png_bytes(fig, width=chart_width, height=chart_height)
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(
            0,
            7,
            _sanitize_pdf_text(section),
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
        )
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(
            0,
            5,
            _sanitize_pdf_text(subtitle),
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
        )
        pdf.ln(1)
        pdf.image(
            io.BytesIO(png),
            x=10,
            w=img_w_mm,
            keep_aspect_ratio=True,
        )

    if not figures:
        pdf.add_page()
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(
            0,
            6,
            _sanitize_pdf_text(
                "No charts available for this view (empty data or missing columns). "
                "Adjust filters or complete schema mapping, then try again."
            ),
        )

    raw = pdf.output()
    if isinstance(raw, (bytes, bytearray)):
        return bytes(raw)
    return str(raw).encode("latin-1")

