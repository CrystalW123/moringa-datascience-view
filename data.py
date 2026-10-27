"""Data generation, cleaning and train/test split shared by every page of the app."""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

SEED = 42
CONTRACT_MAP = {"month-to-month": "Month-to-month", "one year": "One year", "two year": "Two year"}
NUMERIC = [
    "age", "tenure_months", "monthly_charges", "total_charges",
    "support_calls", "avg_monthly_spend", "calls_per_year", "is_new_customer",
]
CATEGORICAL = ["contract", "internet_service", "payment_method"]
TARGET = "churned"


def make_raw(n: int = 5000, seed: int = SEED) -> pd.DataFrame:
    """Simulate a messy subscription-business customer table."""
    rng = np.random.default_rng(seed)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20])
    internet = rng.choice(["DSL", "Fiber", "None"], n, p=[0.35, 0.45, 0.20])
    payment = rng.choice(["Card", "Mobile money", "Bank transfer", "Cash"], n, p=[0.35, 0.30, 0.25, 0.10])
    tenure = np.clip(rng.gamma(2.0, 14.0, n), 1, 72).round().astype(int)
    age = np.clip(rng.normal(41, 13, n), 18, 85).round()
    base = {"DSL": 45, "Fiber": 80, "None": 22}
    monthly = np.clip(np.array([base[i] for i in internet]) + rng.normal(0, 8, n), 10, None).round(2)
    support_calls = rng.poisson(1.2 + 0.9 * (internet == "Fiber"), n)
    total = (monthly * tenure * rng.uniform(0.92, 1.05, n)).round(2)

    logit = (
        -0.6
        + 1.3 * (contract == "Month-to-month")
        - 0.9 * (contract == "Two year")
        - 0.035 * tenure
        + 0.018 * (monthly - 60)
        + 0.28 * support_calls
        + 0.4 * (payment == "Cash")
        - 0.01 * (age - 41)
    )
    churn = rng.random(n) < 1 / (1 + np.exp(-logit))

    df = pd.DataFrame(
        {
            "customer_id": [f"C{100000 + i}" for i in range(n)],
            "age": age,
            "tenure_months": tenure,
            "contract": contract,
            "internet_service": internet,
            "payment_method": payment,
            "monthly_charges": monthly,
            "total_charges": total,
            "support_calls": support_calls,
            "churned": churn.astype(int),
        }
    )
    # Real-world mess: gaps, inconsistent labels, impossible values, duplicates.
    df.loc[rng.random(n) < 0.05, "age"] = np.nan
    df.loc[rng.random(n) < 0.03, "total_charges"] = np.nan
    df.loc[rng.random(n) < 0.02, "contract"] = df["contract"].str.upper()
    df.loc[rng.random(n) < 0.004, "monthly_charges"] = -1
    df = pd.concat([df, df.sample(40, random_state=seed)], ignore_index=True)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


@st.cache_data
def load_raw() -> pd.DataFrame:
    return make_raw()


@st.cache_data
def load_clean() -> pd.DataFrame:
    df = load_raw().drop_duplicates().copy()
    df["contract"] = df["contract"].str.lower().map(CONTRACT_MAP)
    df.loc[df["monthly_charges"] <= 0, "monthly_charges"] = np.nan
    df["avg_monthly_spend"] = df["total_charges"] / df["tenure_months"]
    df["calls_per_year"] = df["support_calls"] / (df["tenure_months"] / 12)
    df["is_new_customer"] = (df["tenure_months"] <= 6).astype(int)
    return df.reset_index(drop=True)


@st.cache_data
def get_split():
    """One shared 80/20 split so every model is judged on the same unseen customers."""
    df = load_clean()
    train, test = train_test_split(df, test_size=0.2, stratify=df[TARGET], random_state=SEED)
    return train.reset_index(drop=True), test.reset_index(drop=True)
