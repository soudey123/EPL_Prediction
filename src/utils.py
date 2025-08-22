import pandas as pd
import numpy as np
import re
from dateutil import parser

def parse_date(x):
    if pd.isna(x): 
        return pd.NaT
    if isinstance(x, pd.Timestamp):
        return x
    try:
        return pd.to_datetime(x, dayfirst=True, errors="coerce")
    except Exception:
        try:
            return parser.parse(str(x), dayfirst=True)
        except Exception:
            return pd.NaT

def safe_div(a, b):
    return a / b if b not in (0, 0.0) else 0.0

_CANON = {
    # Long names -> canonical
    "MANCHESTER UNITED":"Manchester United",
    "MAN UNITED":"Manchester United",
    "MAN UTD":"Manchester United",
    "MANCHESTER UTD":"Manchester United",
    "MANCHESTER CITY":"Manchester City",
    "MAN CITY":"Manchester City",
    "TOTTENHAM HOTSPUR":"Tottenham",
    "SPURS":"Tottenham",
    "WOLVERHAMPTON WANDERERS":"Wolverhampton",
    "WOLVES":"Wolverhampton",
    "NEWCASTLE UTD":"Newcastle United",
    "NEWCASTLE UNITED":"Newcastle United",
    "WEST BROMWICH ALBION":"West Brom",
    "WEST BROM":"West Brom",
    "BRIGHTON & HOVE ALBION":"Brighton",
    "BRIGHTON AND HOVE ALBION":"Brighton",
    "NOTTINGHAM FOREST":"Nott'm Forest",
    "HUDDERSFIELD TOWN":"Huddersfield",
    "LEICESTER CITY":"Leicester",
    "LEEDS UNITED":"Leeds",
    "WEST HAM UNITED":"West Ham",
    "BIRMINGHAM CITY":"Birmingham",
    "CARDIFF CITY":"Cardiff",
    "SWANSEA CITY":"Swansea",
    "HULL CITY":"Hull",
    "NORWICH CITY":"Norwich",
    "STOKE CITY":"Stoke",
    "SUNDERLAND":"Sunderland",
    "EVERTON":"Everton",
    "LIVERPOOL":"Liverpool",
    "CHELSEA":"Chelsea",
    "ARSENAL":"Arsenal",
    "ASTON VILLA":"Aston Villa",
    "CRYSTAL PALACE":"Crystal Palace",
    "FULHAM":"Fulham",
    "BRENTFORD":"Brentford",
    "BOURNEMOUTH":"Bournemouth",
    "SOUTHAMPTON":"Southampton",
    "WATFORD":"Watford",
    "BURNLEY":"Burnley",
    "SHEFFIELD UNITED":"Sheffield United",
    "WIGAN ATHLETIC":"Wigan",
    "READING":"Reading",
    "BLACKPOOL":"Blackpool",
    "DERBY COUNTY":"Derby",
    "PORTSMOUTH":"Portsmouth",
    "MIDDLESBROUGH":"Middlesbrough",
    "CHARLTON ATHLETIC":"Charlton",
    "BOLTON WANDERERS":"Bolton",
    "COVENTRY CITY":"Coventry",
}

_ALLOWED_CHARS = r"[^A-Za-z0-9&\-\.\s']"

def _clean_team(s: str) -> str:
    if not isinstance(s, str):
        return s
    # drop emojis/symbols and normalize spaces
    s2 = re.sub(_ALLOWED_CHARS, " ", s)
    s2 = re.sub(r"\s+", " ", s2).strip()
    # uppercase for mapping, then map to canonical if known
    up = s2.upper()
    if up in _CANON:
        return _CANON[up]
    return s2

def standardize_team_names(df, cols=("HomeTeam","AwayTeam")):
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(str).map(_clean_team)
    return df

def result_label(home_goals, away_goals):
    if home_goals > away_goals: return "H"
    if home_goals < away_goals: return "A"
    return "D"

def brier_score(y_true_proba, y_true_labels, order=("H","D","A")):
    lookup = {lab:i for i, lab in enumerate(order)}
    y_true = np.zeros_like(y_true_proba)
    for i, lab in enumerate(y_true_labels):
        if lab in lookup:
            y_true[i, lookup[lab]] = 1.0
    return np.mean(np.sum((y_true_proba - y_true)**2, axis=1))

def ensure_columns(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df
