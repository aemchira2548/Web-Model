import pandas as pd
import streamlit as st
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).parent


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
