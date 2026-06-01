"""
Dashboard appearance — light and dark backgrounds.

Theme choice is stored in ``st.session_state`` (``state.UI_THEME``). We apply
paired CSS for the page shell *and* widgets (file uploader, buttons, inputs)
so text/background contrast stays correct in both modes.

We do **not** sync Streamlit's localStorage theme — that caused inverted
light/dark when the iframe used the wrong pathname.
"""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from app import state

THEME_LIGHT = "light"
THEME_DARK = "dark"
_VALID_THEMES = frozenset({THEME_LIGHT, THEME_DARK})
_DEFAULT_THEME = THEME_LIGHT

_APPEARANCE_KEY = "ui_appearance_radio"


def normalize_theme(value: str | None) -> str:
    if value in _VALID_THEMES:
        return value
    return _DEFAULT_THEME


def get_theme() -> str:
    return normalize_theme(st.session_state.get(state.UI_THEME, _DEFAULT_THEME))


def set_theme(theme: str) -> None:
    st.session_state[state.UI_THEME] = normalize_theme(theme)


@dataclass(frozen=True)
class PlotTheme:
    title: str
    text: str
    muted: str
    grid: str
    hover_bg: str
    hover_text: str
    gauge_tick: str
    polar_grid: str
    polar_tick: str
    annotation: str


_PLOT_THEMES: dict[str, PlotTheme] = {
    THEME_LIGHT: PlotTheme(
        title="#111827",
        text="#374151",
        muted="#6B7280",
        grid="#F3F4F6",
        hover_bg="#FFFFFF",
        hover_text="#111827",
        gauge_tick="#64748B",
        polar_grid="#E5E7EB",
        polar_tick="#4B5563",
        annotation="#374151",
    ),
    THEME_DARK: PlotTheme(
        title="#F9FAFB",
        text="#E5E7EB",
        muted="#9CA3AF",
        grid="#374151",
        hover_bg="#1F2937",
        hover_text="#F9FAFB",
        gauge_tick="#94A3B8",
        polar_grid="#4B5563",
        polar_tick="#D1D5DB",
        annotation="#E5E7EB",
    ),
}


def plot_theme() -> PlotTheme:
    return _PLOT_THEMES[get_theme()]


def html_muted_color() -> str:
    return "rgba(250, 250, 250, 0.55)" if get_theme() == THEME_DARK else "#6B7280"


def _widget_rules(*, surface: str, surface_alt: str, text: str, muted: str, border: str) -> str:
    """Widget CSS — always set background + text on the same element."""
    return f"""
/* ── Sidebar chrome ── */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div:first-child {{
    background-color: {surface_alt} !important;
}}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] [data-testid="stHeading"],
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] strong,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] {{
    color: {text} !important;
}}
[data-testid="stSidebarNav"] a,
[data-testid="stSidebarNavLink"],
[data-testid="stSidebarNavLink"] span,
[data-testid="stSidebarNavItems"] p {{
    color: {muted} !important;
}}
[data-testid="stSidebarNavLink"][aria-current="page"],
[data-testid="stSidebarNavLink"][aria-current="page"] span {{
    color: {text} !important;
}}

/* ── Main content text ── */
[data-testid="stMain"] h1,
[data-testid="stMain"] h2,
[data-testid="stMain"] h3,
[data-testid="stMain"] h4,
[data-testid="stMain"] [data-testid="stHeading"],
[data-testid="stMain"] [data-testid="stMarkdownContainer"],
[data-testid="stMain"] [data-testid="stMarkdownContainer"] p,
[data-testid="stMain"] [data-testid="stMarkdownContainer"] li,
[data-testid="stMain"] [data-testid="stMarkdownContainer"] strong,
[data-testid="stMain"] [data-testid="stMarkdownContainer"] span,
[data-testid="stMain"] label,
[data-testid="stMain"] [data-testid="stWidgetLabel"] {{
    color: {text} !important;
}}
[data-testid="stCaption"],
[data-testid="stMain"] small {{
    color: {muted} !important;
}}

/* ── Metrics ── */
[data-testid="stMetricLabel"] {{ color: {muted} !important; }}
[data-testid="stMetricValue"] {{ color: {text} !important; }}
[data-testid="stMetricDelta"] {{ color: {muted} !important; }}

/* ── Tabs & expanders ── */
button[data-baseweb="tab"] {{ color: {muted} !important; }}
button[data-baseweb="tab"][aria-selected="true"] {{ color: {text} !important; }}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary span,
[data-testid="stExpanderDetails"] [data-testid="stMarkdownContainer"] {{
    color: {text} !important;
}}

/* ── File uploader (surface + text together) ── */
[data-testid="stFileUploaderDropzone"] {{
    background-color: {surface} !important;
    border-color: {border} !important;
}}
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzoneInstructions"],
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzone"] p,
[data-testid="stFileUploaderDropzone"] small {{
    color: {muted} !important;
}}
[data-testid="stFileUploader"] [data-testid="stWidgetLabel"] {{
    color: {text} !important;
}}

/* ── Buttons ── */
[data-testid="stBaseButton-secondary"],
[data-testid="stBaseButton-secondary"] p,
.stButton > button[kind="secondary"] {{
    background-color: {surface} !important;
    color: {text} !important;
    border-color: {border} !important;
}}
[data-testid="stBaseButton-primary"],
.stButton > button[kind="primary"] {{
    color: #FFFFFF !important;
}}

/* ── Inputs & selects ── */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stDateInput"] input,
textarea,
[data-baseweb="select"] > div,
[data-baseweb="select"] span {{
    background-color: {surface} !important;
    color: {text} !important;
    -webkit-text-fill-color: {text} !important;
    border-color: {border} !important;
}}
[data-testid="stRadio"] label,
[data-testid="stCheckbox"] label,
label[data-baseweb="radio"] {{
    color: {text} !important;
}}

/* ── Data tables (cells only — not toolbar icons) ── */
[data-testid="stDataFrame"],
[data-testid="stDataFrameGlideDataEditor"],
[data-testid="stDataFrameResizable"] {{
    background-color: {surface_alt} !important;
    border-color: {border} !important;
}}
[data-testid="stDataFrame"] [role="gridcell"],
[data-testid="stDataFrame"] [role="columnheader"],
[data-testid="stDataFrameGlideDataEditor"] [role="gridcell"],
[data-testid="stDataFrameGlideDataEditor"] [role="columnheader"] {{
    color: {text} !important;
}}
[data-testid="stDataFrame"] input,
[data-testid="search-input"] {{
    background-color: {surface} !important;
    color: {text} !important;
    -webkit-text-fill-color: {text} !important;
}}

/* Alerts keep Streamlit's own colours */
[data-testid="stAlert"] [data-testid="stMarkdownContainer"],
[data-testid="stAlert"] [data-testid="stMarkdownContainer"] p {{
    color: inherit !important;
}}

hr {{ border-color: {border} !important; }}
"""


_LIGHT_CSS = (
    """
<style>
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
section.main {
    background-color: #FFFFFF !important;
    color: #31333F !important;
}
"""
    + _widget_rules(
        surface="#F0F2F6",
        surface_alt="#F0F2F6",
        text="#31333F",
        muted="rgba(49, 51, 63, 0.65)",
        border="rgba(49, 51, 63, 0.15)",
    )
    + """
</style>
"""
)

# Streamlit keeps its *light* theme internally, so dataframe toolbars render as a
# light-grey pill with grey icons. Forcing icon colour alone (e.g. white) leaves
# white glyphs on a light pill — invisible. Restyle the whole toolbar instead.
_DARK_TOOLBAR_CSS = """
/* Outer wrapper: Streamlit hides it (opacity:0) until you hover the table, which
   made it look "invisible" on dark backgrounds. Keep it always shown. */
[data-testid="stElementToolbar"],
.stElementToolbar {
    opacity: 1 !important;
    top: -2.65rem !important;
}
/* The visible pill is the button container; Streamlit gives it a near-white
   background, so white icons disappeared. Make the pill dark instead. */
[data-testid="stElementToolbarButtonContainer"] {
    background-color: rgba(38, 39, 48, 0.96) !important;
    color: #FAFAFA !important;
    border: 1px solid rgba(250, 250, 250, 0.22) !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.35) !important;
    border-radius: 0.5rem !important;
}
[data-testid="stElementToolbarButtonContainer"] button,
[data-testid="stBaseButton-elementToolbar"],
[data-testid="stElementToolbarButton"] button {
    color: #FAFAFA !important;
    background-color: transparent !important;
    border: none !important;
}
[data-testid="stBaseButton-elementToolbar"]:hover,
[data-testid="stElementToolbarButton"] button:hover {
    background-color: rgba(255, 255, 255, 0.12) !important;
    color: #FAFAFA !important;
}
[data-testid="stElementToolbarButtonIcon"],
[data-testid="stElementToolbarButtonIcon"] span,
[data-testid="stElementToolbarButtonIcon"] svg,
[data-testid="stElementToolbarButtonIcon"] svg path,
[data-testid="stElementToolbar"] svg,
[data-testid="stElementToolbar"] svg path,
[data-testid="stIconMaterial"] {
    color: #FAFAFA !important;
    fill: currentColor !important;
}
[data-testid="stElementToolbar"] img,
[data-testid="stElementToolbarButtonIcon"] img {
    filter: brightness(0) invert(1) !important;
}
"""

_DARK_CSS = (
    """
<style>
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
section.main {
    background-color: #0e1117 !important;
    color: #FAFAFA !important;
}
[data-testid="stHeader"] {
    background-color: rgba(14, 17, 23, 0.95) !important;
}
.js-plotly-plot .plotly .modebar-btn svg {
    fill: rgba(250, 250, 250, 0.85) !important;
}
"""
    + _widget_rules(
        surface="#31333F",
        surface_alt="#262730",
        text="#FAFAFA",
        muted="rgba(250, 250, 250, 0.65)",
        border="rgba(250, 250, 250, 0.20)",
    )
    + _DARK_TOOLBAR_CSS
    + """
</style>
"""
)


def inject_css(theme: str | None = None) -> None:
    """Inject global theme CSS (st.html event container — applies app-wide)."""
    active = normalize_theme(theme or get_theme())
    css = _DARK_CSS if active == THEME_DARK else _LIGHT_CSS
    st.html(css)


def _appearance_options() -> tuple[list[str], dict[str, str]]:
    labels = ["☀️ Light", "🌙 Dark"]
    mapping = {labels[0]: THEME_LIGHT, labels[1]: THEME_DARK}
    return labels, mapping


def render_appearance_toolbar() -> None:
    label_list, labels_to_theme = _appearance_options()
    current = get_theme()
    index = 0 if current == THEME_LIGHT else 1

    _spacer, toolbar = st.columns([5, 3])
    with toolbar:
        chosen_label = st.radio(
            "Appearance",
            label_list,
            index=index,
            horizontal=True,
            key=_APPEARANCE_KEY,
            help="Switch between light and dark backgrounds.",
        )

    chosen = labels_to_theme[chosen_label]
    if chosen != current:
        set_theme(chosen)
        st.rerun()


def apply_page_theme(*, show_toolbar: bool = True) -> None:
    inject_css()
    if show_toolbar:
        render_appearance_toolbar()
