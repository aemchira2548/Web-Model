import pandas as pd
import streamlit as st
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).parent

BADGE_COLORS = {
    "Classification": ("#EEF2FF", "#4F46E5"),
    "Regression": ("#ECFDF5", "#059669"),
    "Clustering": ("#FDF2F8", "#DB2777"),
}

_CSS = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

html, body, [class*="css"] { font-family: "Segoe UI", "Sarabun", sans-serif; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #F5F3FF 0%, #FFFFFF 100%);
    border-right: 1px solid #ECE9FE;
}

div[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #ECE9FE;
    border-radius: 12px;
    padding: 14px 10px;
    box-shadow: 0 2px 6px rgba(108, 92, 231, 0.06);
}

.page-banner {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 22px 28px;
    border-radius: 16px;
    margin-bottom: 22px;
    color: #FFFFFF;
    box-shadow: 0 8px 20px rgba(0,0,0,0.08);
}
.page-banner .icon-badge {
    font-size: 34px;
    background: rgba(255,255,255,0.22);
    border-radius: 14px;
    width: 62px;
    height: 62px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}
.page-banner h1 {
    font-size: 24px;
    margin: 0 0 4px 0;
    color: #FFFFFF;
}
.page-banner p {
    margin: 0;
    opacity: 0.92;
    font-size: 14px;
}
.task-pill {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 600;
    margin-top: 8px;
    background: rgba(255,255,255,0.25);
    color: #FFFFFF;
}

div[data-testid="stVerticalBlockBorderWrapper"]:has(div[data-testid="stPageLink"]) {
    border-radius: 16px !important;
    box-shadow: 0 2px 10px rgba(108, 92, 231, 0.06);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
div[data-testid="stVerticalBlockBorderWrapper"]:has(div[data-testid="stPageLink"]):hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 24px rgba(108, 92, 231, 0.18);
    border-color: #C7C0FB !important;
}

.m-icon {
    font-size: 26px;
    width: 46px;
    height: 46px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 10px;
}
.m-title {
    font-weight: 700;
    font-size: 15.5px;
    color: #1E1B2E;
    margin-bottom: 4px;
}
.m-desc {
    font-size: 13px;
    color: #6B7280;
    margin-bottom: 10px;
    min-height: 36px;
}
.badge-pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.2px;
}

div[data-testid="stPageLink"] {
    background: linear-gradient(120deg, #6C5CE7, #8477F0);
    border-radius: 10px !important;
    padding: 2px 4px;
}
div[data-testid="stPageLink"] p {
    color: #FFFFFF !important;
    font-weight: 600;
    font-size: 13.5px;
}
div[data-testid="stPageLink"]:hover {
    background: linear-gradient(120deg, #5B4BD6, #6C5CE7);
}
</style>
"""


def inject_css():
    st.markdown(_CSS, unsafe_allow_html=True)


def page_header(icon, title, subtitle, task_type="Classification", color="#6C5CE7"):
    inject_css()
    st.markdown(
        f"""
        <div class="page-banner" style="background: linear-gradient(120deg, {color} 0%, {color}CC 100%);">
            <div class="icon-badge">{icon}</div>
            <div>
                <h1>{title}</h1>
                <p>{subtitle}</p>
                <span class="task-pill">{task_type}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_csv(path):
    return pd.read_csv(PROJECT_ROOT / path, encoding="utf-8-sig")


def show_classification_metrics(y_test, y_pred):
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{acc:.2%}")
    c2.metric("Precision", f"{prec:.2%}")
    c3.metric("Recall", f"{rec:.2%}")
    c4.metric("F1-score", f"{f1:.2%}")

    fig, ax = plt.subplots(figsize=(4, 3))
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm).plot(ax=ax, colorbar=False)
    st.pyplot(fig, use_container_width=False)
