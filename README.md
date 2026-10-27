# Customer Churn Lab

An interactive Streamlit app that takes one business problem, **which customers are about to cancel?**, and answers it three ways: data analysis, a rule-based model and machine learning. A final page compares them side by side, in accuracy and in money.

It was built as a teaching companion for the guest lecture *Machine Learning Applications in Data Science*, but it works as a standalone demo for any audience.

## What's inside

| Page | What it shows | Why it matters |
|---|---|---|
| **Home** | The business problem, headline numbers, a preview of the data | Sets the scene before any analysis |
| **1 · Data analysis dashboard** | Filterable KPIs, churn by segment, drivers of churn, data-quality report | Tells you whether a model is worth building and surfaces data problems early |
| **2 · Rule-based model** | Four human-readable rules with adjustable thresholds, results on unseen customers, per-rule performance | Transparent, needs no training data, and sets the baseline a model must beat |
| **3 · Machine learning model** | Choice of three models, adjustable decision threshold, confusion matrix, ROC curves, feature importance, single-customer scorer | Learns patterns humans would not write as rules and gives a risk probability |
| **4 · Compare outputs** | One scoreboard for rules vs. ML, profit under editable campaign costs, CSV download of predictions | The best model is the one that improves the decision, not just the score |

Every page opens with a "Why this matters" box, which doubles as speaker notes.

## Quick start

Requires Python 3.10 or newer.

```bash
# 1. Put all files from this folder in one directory, then:
pip install -r requirements.txt

# 2. Launch the app
streamlit run app.py
```

Streamlit opens the app in your browser (usually at http://localhost:8501). The first load trains the models and takes a few seconds; results are cached afterwards.

## Project structure

```
streamlit_app/
├── app.py            # The Streamlit interface: all five pages
├── data.py           # Data generation, cleaning, feature creation, train/test split
├── models.py         # Rule-based model, ML pipelines, evaluation and profit helpers
├── test_app.py       # Smoke test that opens every page and changes key controls
├── requirements.txt  # Python dependencies
└── README.md
```

## The data

The dataset is **simulated**, so no real customer information is involved and the app runs offline. `data.py` generates 5,000 subscription customers (plus 40 duplicate rows) with columns such as age, tenure, contract type, internet service, payment method, charges and support calls. The target is `churned` (1 = customer left).

Realistic mess is added on purpose so the cleaning step has real work to do:

- duplicate rows
- missing ages and total charges
- the same contract typed two ways (for example `MONTH-TO-MONTH` and `Month-to-month`)
- impossible monthly charges (zero or negative)

Because the data comes from a known generating process, the true drivers of churn (contract type, tenure, support calls, price) are known, which makes it good for teaching. Findings from this data should not be assumed to hold for a real business.

## How the models work

**Rule-based.** A customer is flagged when at least *N* of four rules fire: month-to-month contract, short tenure, many support calls, high monthly charges. You set the thresholds and *N* with sliders.

**Machine learning.** Logistic regression, random forest or gradient boosting, each wrapped in a scikit-learn `Pipeline` that imputes missing values, scales numeric columns and one-hot encodes categories. Fitting the preprocessing inside the pipeline means it learns from training rows only, which avoids test-set leakage.

**Fair comparison.** All models are trained on the same 80% of customers and scored on the same unseen 20% (stratified split, fixed random seed).

**Profit.** The comparison page estimates campaign profit as:

```
profit = (churners contacted × save rate × value of a saved customer) − (customers contacted × cost of an offer)
```

The defaults (offer cost 20, saved-customer value 300, save rate 30%) are **illustrative assumptions**. Change them in the sidebar, and in real work agree them with finance and marketing.

## Using your own data

Replace `load_raw()` and `load_clean()` in `data.py` with code that loads and cleans your table. Keep these column names, or edit the lists at the top of `data.py` and the rule definitions in `models.py`:

- **Numeric:** `age`, `tenure_months`, `monthly_charges`, `total_charges`, `support_calls`, `avg_monthly_spend`, `calls_per_year`, `is_new_customer`
- **Categorical:** `contract`, `internet_service`, `payment_method`
- **Target:** `churned` (0 or 1), plus an ID column `customer_id`

## Testing

```bash
python test_app.py
```

This uses Streamlit's built-in test runner to open every page, switch the ML model, change a rule slider and visit the comparison page. It prints `OK` for each page and stops with an error message if any page raises an exception. It checks that pages run, not that the model results are correct.

## Limitations

- The data is simulated, and the linear model does well partly because the data was generated from a logistic process. On real data, tree-based models often win, so compare rather than assume.
- Feature importance describes what the model relies on, not what causes churn.
- The profit view depends entirely on the assumed costs and save rate. Treat it as a way to reason about trade-offs, not a forecast.
- A real deployment would also need monitoring for data drift, fairness checks across customer groups and privacy safeguards.

## Tech stack

Python, Streamlit, pandas, NumPy, scikit-learn, Matplotlib.