# Quick import check to verify the bundle integrates in your repo.
try:
    import streamlit  # noqa: F401
    from src import tuning, gpt5_scorer, stacking  # noqa: F401
    print("OK: imports succeeded.")
except Exception as e:
    print("Import failure:", e)
