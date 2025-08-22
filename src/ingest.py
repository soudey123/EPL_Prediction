import os, io, zipfile, requests
import pandas as pd
from datetime import datetime
from .utils import parse_date, standardize_team_names, ensure_columns

FOOTBALL_DATA_BASE = "https://www.football-data.co.uk/mmz4281"

def season_code(year_start):
    # e.g., 2015 -> "1516"
    return f"{str(year_start)[-2:]}{str(year_start+1)[-2:]}"

def download_prem_csv(year_start):
    code = season_code(year_start)
    url = f"{FOOTBALL_DATA_BASE}/{code}/E0.csv"
    r = requests.get(url, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Failed to fetch {url} ({r.status_code})")
    df = pd.read_csv(io.BytesIO(r.content))
    df["SeasonStart"] = year_start
    return df

def get_historic_premier_data(start=1993, end=None):
    """
    Grab Premier League historic results from Football-Data.
    start: first season start year (>=1993). If you also have First Division data, merge externally.
    end: last season start (default: current year - 1)
    """
    if end is None:
        end = datetime.now().year - 1
    frames = []
    for y in range(start, end+1):
        try:
            df = download_prem_csv(y)
            frames.append(df)
        except Exception:
            # skip missing seasons if any
            continue
    df = pd.concat(frames, ignore_index=True)
    # Minimal rename
    rename = {"Date":"Date","HomeTeam":"HomeTeam","AwayTeam":"AwayTeam","FTHG":"FTHG","FTAG":"FTAG","FTR":"FTR"}
    df = df.rename(columns=rename)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = standardize_team_names(df, cols=("HomeTeam","AwayTeam"))
    df = df.dropna(subset=["Date","HomeTeam","AwayTeam","FTHG","FTAG"])
    return df[["Date","HomeTeam","AwayTeam","FTHG","FTAG","FTR","SeasonStart"]]

def load_manual_csv(path):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = standardize_team_names(df)
    return df
