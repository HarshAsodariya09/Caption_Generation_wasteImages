"""Light modern UI components and styling for the Waste Caption Curation app."""

import sys
from pathlib import Path
from typing import Optional

_APP_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _APP_DIR.parent
for _p in [str(_PROJECT_ROOT), str(_APP_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import streamlit as st

from taxonomy import BIN_CONFIG, STATUS_CONFIG

LIGHT_THEME_CSS = """
<style>
html, body, [data-testid="stAppViewContainer"] {
    background-color: #F7F8F5 !important;
    color: #1F2937 !important;
    font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
}
[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid #E5E7EB !important;
}
.stApp h1 { font-size: 28px !important; font-weight: 700 !important; color: #1F2937 !important; }
.stApp h2, .stApp h3 { font-size: 18px !important; font-weight: 600 !important; color: #1F2937 !important; }
div[data-testid="stMetric"] {
    background: #FFFFFF !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 14px !important;
    padding: 16px 20px !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05) !important;
}
div[data-testid="stMetric"] label { color: #6B7280 !important; font-size: 0.875rem !important; }
div[data-testid="stMetricValue"] { color: #1F2937 !important; font-size: 1.85rem !important; font-weight: 700 !important; }
.clean-card {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 14px;
    padding: 18px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    margin-bottom: 16px;
}
.pill {
    display: inline-flex;
    align-items: center;
    padding: 3px 10px;
    border-radius: 9999px;
    font-size: 0.8rem;
    font-weight: 600;
}
.bin-badge {
    display: inline-flex;
    align-items: center;
    padding: 6px 14px;
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.95rem;
}
button[kind="primary"] {
    background-color: #16A34A !important;
    border-color: #16A34A !important;
    color: #FFFFFF !important;
    border-radius: 8px !important;
}
button[kind="primary"]:hover { background-color: #15803D !important; border-color: #15803D !important; }
button[kind="secondary"] { border-radius: 8px !important; }
</style>
"""


def inject_theme_css() -> None:
    """Inject clean light styling under 60 lines."""
    st.markdown(LIGHT_THEME_CSS, unsafe_allow_html=True)


def render_status_pill(status: str) -> str:
    """Generate HTML snippet for a status pill."""
    cfg = STATUS_CONFIG.get(status, STATUS_CONFIG["pending"])
    color = cfg["color"]
    bg = cfg["bg"]
    label = cfg["label"]
    return f'<span class="pill" style="color: {color}; background-color: {bg}; border: 1px solid {color}40;">{label}</span>'


def render_bin_badge(bin_name: str, label_override: str = "") -> str:
    """Generate HTML snippet for a bin badge with 12% tint background."""
    cfg = BIN_CONFIG.get(bin_name, BIN_CONFIG["Other"])
    color = cfg["color_hex"]
    bg = cfg["bg_tint"]
    label = label_override if label_override else cfg["badge"]
    return f'<span class="bin-badge" style="color: {color}; background-color: {bg}; border: 1.5px solid {color};">{label}</span>'


def render_empty_state(message: str, button_label: Optional[str] = None, button_key: Optional[str] = None) -> bool:
    """Show a friendly one-sentence empty state with an optional action button."""
    with st.container(border=True):
        st.write(message)
        if button_label:
            return st.button(button_label, key=button_key, type="primary")
    return False
