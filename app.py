"""Airline Passenger Satisfaction - Streamlit demo of 5 classification models.

The app loads the pre-trained models from model/*.joblib and evaluates them on an
uploaded test CSV. Nothing is retrained inside the app.
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
SAMPLE_PATH = BASE_DIR / "test_data.csv"

MODELS = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "KNN": "knn",
    "Naive Bayes": "naive_bayes",
    "Support Vector Machine": "svm",
}
TARGET = "satisfaction"
POSITIVE = "satisfied"
DROP_COLS = ["Unnamed: 0", "id"]  # index / identifier columns (ignored if present)
CLASS_NAMES = ["neutral or dissatisfied", "satisfied"]
METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model(file_stem):
    """Load a saved pipeline (preprocessing + classifier) from disk."""
    return joblib.load(MODEL_DIR / f"{file_stem}.joblib")


def required_features(model):
    """Input columns the saved pipeline was trained on."""
    return list(model.named_steps["prep"].feature_names_in_)


def encode_target(y):
    """Map the target column to 0/1 (1 = satisfied)."""
    if y.dtype == object or str(y.dtype).startswith(("str", "string", "category")):
        return (y.astype(str).str.strip().str.lower() == POSITIVE).astype(int)
    return y.astype(int)


def get_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


@st.cache_data(show_spinner=False)
def evaluate_model(model_name, X, y):
    """Predict with the saved model and compute all metrics (no retraining)."""
    model = load_model(MODELS[model_name])
    y_pred = model.predict(X)
    y_score = get_scores(model, X)
    fpr, tpr, _ = roc_curve(y, y_score)
    return {
        "metrics": {
            "Accuracy": accuracy_score(y, y_pred),
            "AUC": roc_auc_score(y, y_score),
            "Precision": precision_score(y, y_pred),
            "Recall": recall_score(y, y_pred),
            "F1": f1_score(y, y_pred),
            "MCC": matthews_corrcoef(y, y_pred),
        },
        "cm": confusion_matrix(y, y_pred, labels=[0, 1]),
        "report": pd.DataFrame(
            classification_report(y, y_pred, target_names=CLASS_NAMES,
                                  output_dict=True, zero_division=0)).T,
        "roc": (fpr, tpr),
    }


def plot_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    return fig


def plot_roc(fpr, tpr, auc):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_comparison(df):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    df.plot(kind="bar", ax=ax, width=0.8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model comparison on the uploaded test data")
    ax.legend(ncol=6, loc="lower center", bbox_to_anchor=(0.5, -0.38))
    plt.setp(ax.get_xticklabels(), rotation=0)
    fig.tight_layout()
    return fig


def load_test_data(uploaded, use_sample):
    """Return the raw DataFrame from the upload (or the bundled sample)."""
    if uploaded is not None:
        return pd.read_csv(uploaded), uploaded.name
    if use_sample and SAMPLE_PATH.exists():
        return pd.read_csv(SAMPLE_PATH), SAMPLE_PATH.name
    return None, None


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Airline Passenger Satisfaction", layout="wide")
    st.title("Airline Passenger Satisfaction - Classification Models")
    st.write("Upload **test data** (CSV with all input features and the "
             f"`{TARGET}` column), pick a model and see how it performs. "
             "The models are pre-trained and are **not retrained** on your upload.")

    with st.sidebar:
        st.header("Controls")
        uploaded = st.file_uploader("Upload test CSV", type="csv")
        use_sample = False
        if SAMPLE_PATH.exists():
            use_sample = st.checkbox("Use the bundled sample test_data.csv",
                                     value=False, disabled=uploaded is not None)
        model_name = st.selectbox("Select model", list(MODELS))

    df, source = load_test_data(uploaded, use_sample)
    if df is None:
        st.info("Upload a test CSV in the sidebar (or tick the sample option) to begin.")
        return

    # ---- validate and prepare the data ----
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    if TARGET not in df.columns:
        st.error(f"The file must contain the target column `{TARGET}`.")
        return
    model = load_model(MODELS[model_name])
    missing = [c for c in required_features(model) if c not in df.columns]
    if missing:
        st.error(f"Missing required feature columns: {missing}")
        return
    y = encode_target(df[TARGET])
    X = df[required_features(model)]

    st.subheader(f"Dataset: {source}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Features", X.shape[1])
    c3.metric("Satisfied (%)", f"{100 * y.mean():.1f}")
    with st.expander("Preview data"):
        st.dataframe(df.head(20))

    tab_single, tab_compare = st.tabs([f"{model_name} results", "Compare all models"])

    # ---- selected model ----
    with tab_single:
        try:
            with st.spinner(f"Evaluating {model_name}..."):
                res = evaluate_model(model_name, X, y)
        except Exception as e:  # e.g. non-numeric junk in a numeric column
            st.error(f"Could not evaluate the model on this file: {e}")
            return

        cols = st.columns(len(METRIC_NAMES))
        for col, name in zip(cols, METRIC_NAMES):
            col.metric(name, f"{res['metrics'][name]:.4f}")

        left, right = st.columns(2)
        with left:
            fig = plot_confusion_matrix(res["cm"])
            st.pyplot(fig)
            plt.close(fig)
        with right:
            fig = plot_roc(*res["roc"], res["metrics"]["AUC"])
            st.pyplot(fig)
            plt.close(fig)

        st.subheader("Classification report")
        st.dataframe(res["report"].round(4))

    # ---- all models ----
    with tab_compare:
        try:
            with st.spinner("Evaluating all five models (SVM and KNN can take a moment)..."):
                rows = {name: evaluate_model(name, X, y)["metrics"] for name in MODELS}
        except Exception as e:
            st.error(f"Could not evaluate the models on this file: {e}")
            return
        comp = pd.DataFrame(rows).T[METRIC_NAMES]
        comp.index.name = "ML Model Name"
        st.dataframe(comp.round(4))
        best = comp["MCC"].idxmax()
        st.success(f"Best model by MCC on this data: **{best}** "
                   f"(MCC = {comp.loc[best, 'MCC']:.4f})")
        fig = plot_comparison(comp)
        st.pyplot(fig)
        plt.close(fig)


if __name__ == "__main__":
    main()
