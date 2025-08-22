import os
import pandas as pd
import numpy as np
import requests
from .utils import parse_date, standardize_team_names, ensure_columns
from .features import build_features

API_BASE = "https://v3.football.api-sports.io"

class FixturesFetchError(RuntimeError):
    pass

def fetch_fixtures_api(season=2025, league=39, api_key=None):
    if api_key is None:
        api_key = os.getenv("API_FOOTBALL_KEY")
    if not api_key:
        raise FixturesFetchError("API_FOOTBALL_KEY not set. Add it to your .env or Replit Secrets.")
    headers = {"x-apisports-key": api_key}
    out = []
    page = 1
    try:
        while True:
            params = {"league": league, "season": season, "page": page}
            r = requests.get(f"{API_BASE}/fixtures", headers=headers, params=params, timeout=30)
            r.raise_for_status()
            js = r.json()
            errs = js.get("errors") or {}
            if errs:
                raise FixturesFetchError(f"API error: {errs}")
            for item in js.get("response", []):
                home = item["teams"]["home"]["name"]
                away = item["teams"]["away"]["name"]
                date = item["fixture"]["date"]
                out.append({"Date": pd.to_datetime(date, errors="coerce"), "HomeTeam":home, "AwayTeam":away})
            total_pages = js.get("paging", {}).get("total", 1) or 1
            if page >= total_pages:
                break
            page += 1
    except Exception as e:
        raise FixturesFetchError(f"Failed to fetch fixtures: {e}")
    if len(out) == 0:
        raise FixturesFetchError("API returned 0 fixtures. Check key/quota or try Upload CSV/ICS.")
    df = pd.DataFrame(out, columns=["Date","HomeTeam","AwayTeam"]).sort_values("Date")
    df = standardize_team_names(df)
    return df

def prepare_prediction_rows(historic_matches, fixtures_df):
    # Ensure team names are standardized on both sides to align history with future fixtures
    fixtures_df = standardize_team_names(fixtures_df.copy())
    historic_matches = standardize_team_names(historic_matches.copy(), cols=("HomeTeam","AwayTeam"))
    # Engineer features up to the last historic date; append fixtures (with NaN scores) then rebuild features.
    hist_feats = build_features(historic_matches)
    hist_feats = hist_feats.sort_values("Date")
    future = fixtures_df.copy()
    future["FTHG"] = np.nan; future["FTAG"] = np.nan; future["FTR"] = np.nan
    combined = pd.concat([historic_matches[["Date","HomeTeam","AwayTeam","FTHG","FTAG","FTR"]], future], ignore_index=True)
    feats = build_features(combined)
    future_rows = feats[feats["FTR"].isna()].copy()
    return hist_feats.dropna(subset=["label"]), future_rows
