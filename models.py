"""Rule-based and machine learning models, plus shared evaluation helpers."""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data import CATEGORICAL, NUMERIC, SEED, TARGET, get_split

ML_MODELS = {
    "Logistic Regression": lambda: LogisticRegression(max_iter=1000),
    "Random Forest": lambda: RandomForestClassifier(
        n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=SEED
    ),
    "Gradient Boosting": lambda: HistGradientBoostingClassifier(random_state=SEED),
}

DEFAULT_RULES = {"tenure_max": 12, "calls_min": 3, "charge_min": 70, "min_rules": 2}


# ---------------------------------------------------------------- rule-based
def rule_flags(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Each column is one human-readable rule: True when the rule fires."""
    return pd.DataFrame(
        {
            "Month-to-month contract": df["contract"] == "Month-to-month",
            f"Tenure <= {cfg['tenure_max']} months": df["tenure_months"] <= cfg["tenure_max"],
            f"{cfg['calls_min']}+ support calls": df["support_calls"] >= cfg["calls_min"],
            f"Monthly charges >= {cfg['charge_min']}": df["monthly_charges"] >= cfg["charge_min"],
        },
        index=df.index,
    )


def rule_predict(df: pd.DataFrame, cfg: dict):
    """Return (rules fired per customer, 0/1 prediction)."""
    fired = rule_flags(df, cfg).sum(axis=1)
    return fired, (fired >= cfg["min_rules"]).astype(int)


# ------------------------------------------------------------ machine learning
def build_pipeline(name: str) -> Pipeline:
    numeric_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    cat_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    prep = ColumnTransformer([("num", numeric_pipe, NUMERIC), ("cat", cat_pipe, CATEGORICAL)])
    return Pipeline([("prep", prep), ("model", ML_MODELS[name]())])


@st.cache_resource
def train_model(name: str) -> Pipeline:
    train, _ = get_split()
    return build_pipeline(name).fit(train[NUMERIC + CATEGORICAL], train[TARGET])


@st.cache_data
def ml_probabilities(name: str) -> np.ndarray:
    _, test = get_split()
    return train_model(name).predict_proba(test[NUMERIC + CATEGORICAL])[:, 1]


@st.cache_data
def importance(name: str) -> pd.Series:
    _, test = get_split()
    X, y = test[NUMERIC + CATEGORICAL], test[TARGET]
    res = permutation_importance(
        train_model(name), X, y, scoring="roc_auc", n_repeats=5, random_state=SEED, n_jobs=-1
    )
    return pd.Series(res.importances_mean, index=X.columns).sort_values()


# ------------------------------------------------------------------ evaluation
def score(y_true, y_pred) -> dict:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }


def campaign_profit(y_true, y_pred, cost: float, value: float, save_rate: float) -> float:
    """Profit of contacting every flagged customer with a retention offer."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    hit = ((y_pred == 1) & (y_true == 1)).sum()
    return float(hit * save_rate * value - (y_pred == 1).sum() * cost)
