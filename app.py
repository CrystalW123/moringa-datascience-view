"""
Customer Churn Lab: data analysis, rule-based model and machine learning in one app.

Run:  streamlit run app.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay, roc_auc_score, roc_curve

from data import CATEGORICAL, NUMERIC, TARGET, get_split, load_clean, load_raw
from models import (
    DEFAULT_RULES,
    ML_MODELS,
    campaign_profit,
    importance,
    ml_probabilities,
    rule_flags,
    rule_predict,
    score,
    train_model,
)

st.set_page_config(page_title="Customer Churn Lab", page_icon="📊", layout="wide")

ACCENT, TEAL, INK = "#E2672C", "#17687A", "#0E1A2B"


# ---------------------------------------------------------------- small helpers
def why(title: str, body: str) -> None:
    with st.expander(f"Why this matters: {title}", expanded=True):
        st.markdown(body)


def bar_chart(series: pd.Series, title: str, pct: bool = True, color: str = TEAL):
    fig, ax = plt.subplots(figsize=(5, 3.2))
    series.plot.bar(ax=ax, color=color, rot=0)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("")
    if pct:
        ax.set_ylim(0, min(1, series.max() * 1.25))
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.0%}" if pct else f"{p.get_height():,.0f}",
            (p.get_x() + p.get_width() / 2, p.get_height()),
            ha="center", va="bottom", fontsize=9,
        )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def show(fig) -> None:
    st.pyplot(fig)
    plt.close(fig)


def confusion_fig(y_true, y_pred, title: str):
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["Stayed", "Churned"], cmap="Blues", ax=ax, colorbar=False
    )
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    return fig


def metric_row(m: dict, extra: dict | None = None) -> None:
    items = {**m, **(extra or {})}
    for col, (k, v) in zip(st.columns(len(items)), items.items()):
        col.metric(k, f"{v:.1%}" if isinstance(v, float) and v <= 1 else f"{v:,.0f}")


# ------------------------------------------------------------------------ home
def page_home():
    st.title("📊 Customer Churn Lab")
    st.caption("From raw data to a business decision, three ways: analysis, rules and machine learning.")
    raw = load_raw()
    df = load_clean()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{len(df):,}")
    c2.metric("Churn rate", f"{df[TARGET].mean():.1%}")
    c3.metric("Columns", raw.shape[1])
    c4.metric("Data problems fixed", f"{raw.duplicated().sum() + raw.isna().sum().sum():,}")

    st.markdown(
        """
### The business problem
A subscription company loses customers every month. The retention team can only contact
a limited number of people, so **who should they call first?**

### How this app answers it
| Step | Page | Question it answers |
|---|---|---|
| 1 | **Data analysis dashboard** | What is happening, and what seems to drive churn? |
| 2 | **Rule-based model** | Can simple, human-written rules find the at-risk customers? |
| 3 | **Machine learning model** | Can a model learn patterns that rules miss? |
| 4 | **Compare outputs** | Which approach is better, in accuracy *and* in money? |

Use the sidebar to move between pages. The data is **simulated** (no real customers),
with realistic mess added on purpose: duplicates, gaps, inconsistent labels and impossible values.
"""
    )
    st.subheader("A look at the data")
    st.dataframe(df.drop(columns=["avg_monthly_spend", "calls_per_year", "is_new_customer"]).head(10))


# ------------------------------------------------------------------- dashboard
def page_dashboard():
    st.title("1 · Data analysis dashboard")
    why(
        "data analysis comes first",
        """
- **It tells you whether a model is even worth building.** If nothing separates churners from stayers, no algorithm will rescue you.
- **It produces hypotheses** that guide rules and features (for example, "short tenure looks risky").
- **It catches data problems early.** Duplicates, missing values and impossible numbers quietly damage every model built on top.
- **It is how you talk to the business.** A clear chart convinces a manager faster than a model score.
""",
    )
    df = load_clean()

    st.sidebar.header("Dashboard filters")
    contracts = st.sidebar.multiselect("Contract", sorted(df.contract.unique()), default=sorted(df.contract.unique()))
    nets = st.sidebar.multiselect(
        "Internet service", sorted(df.internet_service.unique()), default=sorted(df.internet_service.unique())
    )
    lo, hi = int(df.tenure_months.min()), int(df.tenure_months.max())
    tenure = st.sidebar.slider("Tenure (months)", lo, hi, (lo, hi))
    view = df[df.contract.isin(contracts) & df.internet_service.isin(nets) & df.tenure_months.between(*tenure)]
    if view.empty:
        st.warning("No customers match these filters.")
        return

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Customers", f"{len(view):,}")
    k2.metric("Churn rate", f"{view[TARGET].mean():.1%}", f"{view[TARGET].mean() - df[TARGET].mean():+.1%} vs all", delta_color="inverse")
    k3.metric("Avg tenure", f"{view.tenure_months.mean():.1f} months")
    k4.metric("Avg monthly charge", f"{view.monthly_charges.mean():.0f}")

    tab1, tab2, tab3 = st.tabs(["Churn by segment", "What drives churn?", "Data quality"])

    with tab1:
        a, b, c = st.columns(3)
        with a:
            show(bar_chart(view.groupby("contract")[TARGET].mean(), "Churn rate by contract"))
        with b:
            show(bar_chart(view.groupby("internet_service")[TARGET].mean(), "Churn rate by internet service"))
        with c:
            show(bar_chart(view.groupby("payment_method")[TARGET].mean(), "Churn rate by payment method"))
        worst = view.groupby("contract")[TARGET].mean().idxmax()
        st.info(f"Highest-risk contract type in this view: **{worst}** "
                f"({view.groupby('contract')[TARGET].mean().max():.0%} churn).")

    with tab2:
        a, b = st.columns(2)
        with a:
            fig, ax = plt.subplots(figsize=(5.5, 3.4))
            for label, grp in view.groupby(TARGET):
                ax.hist(grp.tenure_months, bins=24, alpha=0.6, color=ACCENT if label else TEAL,
                        label="Churned" if label else "Stayed")
            ax.set(title="Tenure: churners leave early", xlabel="Tenure (months)")
            ax.legend()
            ax.spines[["top", "right"]].set_visible(False)
            fig.tight_layout()
            show(fig)
        with b:
            calls = view.groupby(view.support_calls.clip(upper=7))[TARGET].mean()
            fig, ax = plt.subplots(figsize=(5.5, 3.4))
            ax.plot(calls.index, calls.values, marker="o", color=ACCENT)
            ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
            ax.set(title="More support calls, more churn", xlabel="Support calls (7 = 7 or more)")
            ax.spines[["top", "right"]].set_visible(False)
            fig.tight_layout()
            show(fig)
        a, b = st.columns(2)
        with a:
            fig, ax = plt.subplots(figsize=(5.5, 3.4))
            bins = pd.cut(view.monthly_charges, 8)
            rate = view.groupby(bins, observed=True)[TARGET].mean()
            ax.bar(range(len(rate)), rate.values, color=TEAL)
            ax.set_xticks(range(len(rate)), [f"{int(i.left)}-{int(i.right)}" for i in rate.index], fontsize=8)
            ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
            ax.set(title="Churn rate by monthly charge band", xlabel="Monthly charges")
            ax.spines[["top", "right"]].set_visible(False)
            fig.tight_layout()
            show(fig)
        with b:
            corr = view[NUMERIC + [TARGET]].corr()[TARGET].drop(TARGET).sort_values()
            fig, ax = plt.subplots(figsize=(5.5, 3.4))
            ax.barh(corr.index, corr.values, color=[ACCENT if v > 0 else TEAL for v in corr.values])
            ax.set(title="Correlation with churn")
            ax.spines[["top", "right"]].set_visible(False)
            fig.tight_layout()
            show(fig)
        st.caption("Correlation shows association, not cause. Orange = linked to more churn, teal = linked to less.")

    with tab3:
        raw = load_raw()
        a, b, c, d = st.columns(4)
        a.metric("Duplicate rows removed", int(raw.duplicated().sum()))
        b.metric("Missing ages", int(raw.age.isna().sum()))
        c.metric("Missing total charges", int(raw.total_charges.isna().sum()))
        d.metric("Impossible monthly charges", int((raw.monthly_charges <= 0).sum()))
        st.markdown("**Contract labels before cleaning** (the same category typed two ways):")
        st.dataframe(raw.contract.value_counts().rename("rows"))
        st.markdown(
            "Cleaning decisions: duplicates dropped, labels standardised, impossible values set to missing. "
            "Remaining gaps are filled **inside the model pipeline** so the test set never leaks into training."
        )


# ------------------------------------------------------------------ rule-based
def page_rules():
    st.title("2 · Rule-based model")
    why(
        "rules come before machine learning",
        """
- **Transparent.** Anyone can read the rules, challenge them and explain a decision to a customer or regulator.
- **No training data needed.** You can launch on day one from expert knowledge.
- **A strong baseline.** If machine learning cannot clearly beat sensible rules, it may not be worth its extra cost and complexity.
- **Easy to maintain.** The business can change a threshold without a data scientist.

The trade-off: rules are written by humans, so they miss patterns nobody thought of and do not weigh evidence.
""",
    )
    cfg = st.session_state.get("rule_cfg", DEFAULT_RULES)
    st.subheader("Write the rules")
    c1, c2, c3, c4 = st.columns(4)
    new_cfg = {
        "tenure_max": c1.slider("Rule 2: tenure at most (months)", 1, 36, cfg["tenure_max"]),
        "calls_min": c2.slider("Rule 3: support calls at least", 1, 8, cfg["calls_min"]),
        "charge_min": c3.slider("Rule 4: monthly charges at least", 20, 120, cfg["charge_min"]),
        "min_rules": c4.slider("Flag a customer when this many rules fire", 1, 4, cfg["min_rules"]),
    }
    st.session_state["rule_cfg"] = new_cfg
    st.caption("Rule 1 is fixed: the customer is on a month-to-month contract.")

    _, test = get_split()
    fired, pred = rule_predict(test, new_cfg)
    y = test[TARGET]

    st.subheader("Results on unseen customers")
    metric_row(score(y, pred), {"Customers flagged": int(pred.sum())})

    a, b = st.columns(2)
    with a:
        show(confusion_fig(y, pred, "Confusion matrix: rule-based"))
    with b:
        rate = pd.DataFrame({"fired": fired, "churn": y}).groupby("fired")["churn"].agg(["mean", "size"])
        show(bar_chart(rate["mean"], "Churn rate by number of rules fired"))
        st.caption("A good rule set shows churn rising steadily as more rules fire.")

    st.subheader("How well does each rule work alone?")
    flags = rule_flags(test, new_cfg)
    rows = []
    for col in flags:
        m = flags[col]
        rows.append(
            {
                "Rule": col,
                "Customers it fires for": int(m.sum()),
                "Churn rate when it fires": y[m].mean() if m.any() else np.nan,
                "Share of all churners caught": (m & (y == 1)).sum() / (y == 1).sum(),
            }
        )
    st.dataframe(
        pd.DataFrame(rows).style.format(
            {"Churn rate when it fires": "{:.1%}", "Share of all churners caught": "{:.1%}"}
        ),
        hide_index=True,
    )

    st.subheader("Highest-risk customers by rules")
    out = test.assign(rules_fired=fired, flagged=pred)
    st.dataframe(
        out.sort_values("rules_fired", ascending=False)[
            ["customer_id", "contract", "tenure_months", "support_calls", "monthly_charges", "rules_fired", "churned"]
        ].head(10),
        hide_index=True,
    )


# ---------------------------------------------------------------------- ML
def page_ml():
    st.title("3 · Machine learning model")
    why(
        "learn the patterns instead of guessing them",
        """
- **It weighs many signals together.** A model combines contract, tenure, calls, price and more in proportions learned from data.
- **It gives a probability, not just yes or no,** so you can rank customers and choose how many to contact.
- **It can find patterns humans miss,** and it improves as more data arrives.
- **It must be tested honestly.** We train on 80% of customers and judge only on the 20% the model never saw.
""",
    )
    cfg = st.session_state.get("ml_cfg", {"model": "Logistic Regression", "threshold": 0.5})
    names = list(ML_MODELS)
    c1, c2 = st.columns([1, 2])
    name = c1.selectbox("Model", names, index=names.index(cfg["model"]))
    threshold = c2.slider("Decision threshold (flag customers with churn probability at least)", 0.05, 0.95, float(cfg["threshold"]), 0.05)
    st.session_state["ml_cfg"] = {"model": name, "threshold": threshold}

    with st.spinner("Training model..."):
        pipe = train_model(name)
    _, test = get_split()
    y = test[TARGET]
    proba = ml_probabilities(name)
    pred = (proba >= threshold).astype(int)

    st.subheader("Results on unseen customers")
    metric_row(score(y, pred), {"ROC AUC": roc_auc_score(y, proba), "Customers flagged": int(pred.sum())})

    a, b, c = st.columns(3)
    with a:
        show(confusion_fig(y, pred, f"Confusion matrix: {name}"))
    with b:
        fig, ax = plt.subplots(figsize=(4.2, 3.6))
        for n in names:
            p = ml_probabilities(n)
            fpr, tpr, _ = roc_curve(y, p)
            ax.plot(fpr, tpr, lw=3 if n == name else 1.2, label=f"{n} ({roc_auc_score(y, p):.2f})")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax.set(title="ROC curves", xlabel="False positive rate", ylabel="True positive rate")
        ax.legend(fontsize=7, loc="lower right")
        fig.tight_layout()
        show(fig)
    with c:
        fig, ax = plt.subplots(figsize=(4.2, 3.6))
        ax.hist(proba[y == 0], bins=25, alpha=0.6, color=TEAL, label="Stayed")
        ax.hist(proba[y == 1], bins=25, alpha=0.6, color=ACCENT, label="Churned")
        ax.axvline(threshold, color="k", ls="--")
        ax.set(title="Predicted churn probability", xlabel="Probability")
        ax.legend(fontsize=8)
        fig.tight_layout()
        show(fig)

    st.subheader("What is the model relying on?")
    imp = importance(name)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.barh(imp.index, imp.values, color=TEAL)
    ax.set(title="Permutation importance (drop in AUC when a column is shuffled)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    show(fig)

    st.subheader("Try it: score one customer")
    with st.form("customer"):
        a, b, c, d = st.columns(4)
        age = a.number_input("Age", 18, 85, 35)
        tenure = b.number_input("Tenure (months)", 1, 72, 6)
        contract = c.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        internet = d.selectbox("Internet", ["DSL", "Fiber", "None"])
        a, b, c, d = st.columns(4)
        payment = a.selectbox("Payment", ["Card", "Mobile money", "Bank transfer", "Cash"])
        monthly = b.number_input("Monthly charges", 10.0, 150.0, 80.0)
        calls = c.number_input("Support calls", 0, 15, 4)
        go = d.form_submit_button("Predict")
    if go:
        row = pd.DataFrame(
            [
                {
                    "age": age, "tenure_months": tenure, "contract": contract, "internet_service": internet,
                    "payment_method": payment, "monthly_charges": monthly, "total_charges": monthly * tenure,
                    "support_calls": calls, "avg_monthly_spend": monthly, "calls_per_year": calls / (tenure / 12),
                    "is_new_customer": int(tenure <= 6),
                }
            ]
        )
        p = float(pipe.predict_proba(row[NUMERIC + CATEGORICAL])[0, 1])
        rule_cfg = st.session_state.get("rule_cfg", DEFAULT_RULES)
        _, rule_pred = rule_predict(row, rule_cfg)
        m1, m2 = st.columns(2)
        m1.metric("Churn probability (ML)", f"{p:.0%}", "flag for retention" if p >= threshold else "no action", delta_color="off")
        m2.metric("Rule-based verdict", "Flag" if rule_pred.iloc[0] else "No action")
        st.progress(min(max(p, 0.0), 1.0))


# --------------------------------------------------------------------- compare
def page_compare():
    st.title("4 · Compare outputs")
    why(
        "the best model is the one that helps the decision",
        """
- **Same customers, same scoreboard.** Both approaches are judged on the identical unseen test set.
- **Accuracy is not the goal; money and missed churners are.** A cheap offer to many customers can beat a precise but narrow list.
- **It shows whether complexity pays off.** If rules are nearly as good, ship the rules.
""",
    )
    rule_cfg = st.session_state.get("rule_cfg", DEFAULT_RULES)
    ml_cfg = st.session_state.get("ml_cfg", {"model": "Logistic Regression", "threshold": 0.5})
    _, test = get_split()
    y = test[TARGET]

    st.sidebar.header("Campaign economics")
    cost = st.sidebar.number_input("Cost of one retention offer", 1.0, 500.0, 20.0)
    value = st.sidebar.number_input("Value of a saved customer", 10.0, 5000.0, 300.0)
    save_rate = st.sidebar.slider("Share of contacted churners retained", 0.05, 1.0, 0.30, 0.05)
    st.sidebar.caption("Illustrative assumptions. In real work, agree these with finance and marketing.")

    _, rule_pred = rule_predict(test, rule_cfg)
    fired, _ = rule_predict(test, rule_cfg)
    name, thr = ml_cfg["model"], ml_cfg["threshold"]
    proba = ml_probabilities(name)
    ml_pred = (proba >= thr).astype(int)

    grid = np.arange(0.05, 0.96, 0.05)
    profits = [campaign_profit(y, (proba >= t).astype(int), cost, value, save_rate) for t in grid]
    best_t = float(grid[int(np.argmax(profits))])
    best_pred = (proba >= best_t).astype(int)

    entries = {
        "Rule-based": rule_pred,
        f"{name} (threshold {thr:.2f})": ml_pred,
        f"{name} (profit-best {best_t:.2f})": best_pred,
    }
    rows = []
    for label, p in entries.items():
        rows.append(
            {"Approach": label, **score(y, p), "Customers contacted": int(p.sum()),
             "Estimated profit": campaign_profit(y, p, cost, value, save_rate)}
        )
    table = pd.DataFrame(rows).set_index("Approach")
    st.subheader("Scoreboard")
    st.dataframe(
        table.style.format({"Accuracy": "{:.1%}", "Precision": "{:.1%}", "Recall": "{:.1%}", "F1": "{:.1%}",
                            "Customers contacted": "{:,.0f}", "Estimated profit": "{:,.0f}"})
        .highlight_max(subset=["Estimated profit"], color="#FBE3D6")
    )
    st.caption(f"Rule settings and ML model/threshold come from the previous pages "
               f"(rule-based: {rule_cfg['min_rules']} of 4 rules).")

    a, b = st.columns(2)
    with a:
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        for label, s in [("Rule-based (rules fired)", fired), (name, proba)]:
            fpr, tpr, _ = roc_curve(y, s)
            ax.plot(fpr, tpr, lw=2, label=f"{label}: AUC {roc_auc_score(y, s):.2f}")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax.set(title="Ranking quality (ROC)", xlabel="False positive rate", ylabel="True positive rate")
        ax.legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        show(fig)
    with b:
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        ax.plot(grid, profits, marker="o", color=ACCENT)
        ax.axvline(best_t, ls="--", color="gray")
        ax.set(title=f"{name}: profit vs threshold", xlabel="Threshold", ylabel="Estimated profit")
        fig.tight_layout()
        show(fig)

    best_row = table["Estimated profit"].idxmax()
    st.success(
        f"**Takeaway:** under these assumptions, *{best_row}* earns the most "
        f"({table['Estimated profit'].max():,.0f}). The best machine learning threshold is about {best_t:.2f}, "
        f"which means contacting {int(best_pred.sum())} of {len(test)} customers."
    )

    st.subheader("Download the outputs")
    out = test[["customer_id", "contract", "tenure_months", "support_calls", "monthly_charges", TARGET]].assign(
        rules_fired=fired, rule_flag=rule_pred, ml_probability=proba.round(4), ml_flag=ml_pred
    )
    st.download_button("Download predictions (CSV)", out.to_csv(index=False), "churn_predictions.csv", "text/csv")


# ------------------------------------------------------------------------ main
PAGES = {
    "Home": page_home,
    "1 · Data analysis dashboard": page_dashboard,
    "2 · Rule-based model": page_rules,
    "3 · Machine learning model": page_ml,
    "4 · Compare outputs": page_compare,
}
choice = st.sidebar.radio("Go to", list(PAGES))
st.sidebar.divider()
PAGES[choice]()
