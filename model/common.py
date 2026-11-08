"""Shared data loading, preprocessing and evaluation helpers for all 5 models."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = ROOT / "data" / "train.csv"
TEST_PATH = ROOT / "test_data.csv"
SAVE_DIR = Path(__file__).resolve().parent  # trained models are saved in model/

TARGET = "satisfaction"
POSITIVE = "satisfied"          # positive class = 1
DROP_COLS = ["Unnamed: 0", "id"]  # index / identifier columns, not predictive
RANDOM_STATE = 42


def load_data():
    """Return X_train, y_train, X_test, y_test (target encoded as 1 = satisfied)."""
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    out = []
    for df in (train, test):
        df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
        y = (df[TARGET] == POSITIVE).astype(int)
        X = df.drop(columns=[TARGET])
        out += [X, y]
    X_train, y_train, X_test, y_test = out
    return X_train, y_train, X_test, y_test


def build_preprocessor(X):
    """Impute + scale numeric columns, one-hot encode categorical columns."""
    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler())])
    categorical = Pipeline([("onehot", OneHotEncoder(drop="if_binary",
                                                     handle_unknown="ignore"))])
    return ColumnTransformer([("num", numeric, num_cols),
                              ("cat", categorical, cat_cols)])


def get_scores(model, X):
    """Continuous score for the positive class (needed for AUC)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def evaluate(model, X_test, y_test):
    """Compute the 6 required metrics."""
    y_pred = model.predict(X_test)
    y_score = get_scores(model, X_test)
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_score),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }


def run_and_save(name, filename, model, X_train, y_train, X_test, y_test):
    """Fit, evaluate, print and persist the model + its metrics."""
    model.fit(X_train, y_train)
    metrics = evaluate(model, X_test, y_test)
    print(f"\n=== {name} ===")
    for k, v in metrics.items():
        print(f"{k:<10}: {v:.4f}")
    joblib.dump(model, SAVE_DIR / f"{filename}.joblib", compress=3)
    with open(SAVE_DIR / f"{filename}_metrics.json", "w") as f:
        json.dump({"model": name, **{k: float(v) for k, v in metrics.items()}}, f, indent=2)
    return metrics
