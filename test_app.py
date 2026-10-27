"""Smoke test: open every page and fail if any raises an exception."""
from streamlit.testing.v1 import AppTest

PAGES = [
    "Home",
    "1 · Data analysis dashboard",
    "2 · Rule-based model",
    "3 · Machine learning model",
    "4 · Compare outputs",
]

at = AppTest.from_file("app.py", default_timeout=180).run()
for page in PAGES:
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, f"{page}: {[e.value for e in at.exception]}"
    print(f"OK  {page}: {len(at.metric)} metrics, {len(at.dataframe)} tables")

# Exercise controls: change a model and the rule settings, then compare.
at.sidebar.radio[0].set_value("3 · Machine learning model").run()
at.selectbox[0].set_value("Random Forest").run()
assert not at.exception, [e.value for e in at.exception]
at.sidebar.radio[0].set_value("2 · Rule-based model").run()
at.slider[3].set_value(3).run()
assert not at.exception, [e.value for e in at.exception]
at.sidebar.radio[0].set_value("4 · Compare outputs").run()
assert not at.exception, [e.value for e in at.exception]
print("OK  control changes carry through to Compare")
