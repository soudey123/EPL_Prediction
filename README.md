# EPL Predictor (2025–26)

A Streamlit web app to predict Premier League match outcomes (Home/Draw/Away) for the 2025–26 season.
It combines historical data (Premier League & old First Division via Football-Data), team Elo,
recent form, head-to-head stats, and gradient boosting models with probability calibration.

## Features
- One-click historical data ingestion from Football-Data CSVs (or upload your own).
- Automated feature engineering: rolling form, goal stats, head-to-head aggregates, Elo.
- Train/validate calibrated ML models (Logistic Regression, Gradient Boosting).
- Predict upcoming fixtures for the 2025–26 season (from API-Football or uploaded CSV).
- Explain predictions (feature importance) and evaluate with Brier score & Log Loss.
- Export predictions as CSV.

## Quick Start
1) Create a virtual env (Python 3.10+ recommended) and install requirements:
```bash
pip install -r requirements.txt
```

2) (Optional, for live fixtures) Get an API key at https://www.api-football.com/ (free plan ok).
Set an environment variable before running the app:
```bash
export API_FOOTBALL_KEY=YOUR_KEY
```

3) Run:
```bash
streamlit run streamlit_app.py
```

## Data Sources (supported)
- Football-Data.co.uk CSVs (historic results & some odds): https://www.football-data.co.uk/  (E0 = Premier League)
- API-Football fixtures for 2025–26 (league=39, season=2025)
- Manual CSV upload (columns: Date, HomeTeam, AwayTeam; optional: neutral flag)

## Notes
- The “125 years” of top-flight history is approximated by including the pre-Premier-League First Division where available.
- Out-of-the-box, the app prioritizes bookmaker-independent features (form/Elo/head-to-head). If you include odds in training,
  the model may improve but ensure you won’t use odds at prediction time to avoid leakage.
