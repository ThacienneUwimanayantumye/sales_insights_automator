"""Tests for dashboard appearance theme helpers."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.theme import (
    THEME_DARK,
    THEME_LIGHT,
    _PLOT_THEMES,
    normalize_theme,
)


class TestTheme:
    def test_normalize_defaults_to_light(self):
        assert normalize_theme(None) == THEME_LIGHT
        assert normalize_theme("invalid") == THEME_LIGHT
        assert normalize_theme("") == THEME_LIGHT

    def test_normalize_accepts_valid(self):
        assert normalize_theme(THEME_DARK) == THEME_DARK
        assert normalize_theme(THEME_LIGHT) == THEME_LIGHT

    def test_light_and_dark_plot_palettes_differ(self):
        assert _PLOT_THEMES[THEME_LIGHT].title != _PLOT_THEMES[THEME_DARK].title
        assert _PLOT_THEMES[THEME_LIGHT].grid != _PLOT_THEMES[THEME_DARK].grid

    def test_html_muted_color_is_hex_or_rgba(self):
        from app.theme import html_muted_color

        c = html_muted_color()
        assert c.startswith("#") or c.startswith("rgba")

    def test_dark_css_styles_dataframe_toolbar(self):
        from app.theme import _DARK_CSS

        assert '[data-testid="stElementToolbar"]' in _DARK_CSS
        assert "background-color: rgba(38, 39, 48" in _DARK_CSS
        assert "data-sia-theme" not in _DARK_CSS
