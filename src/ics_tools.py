import re
from datetime import timezone
import pandas as pd
from ics import Calendar

def _parse_match_title(title: str):
    """Extract (home, away, competition) from common PL/club ICS titles."""
    if not title:
        return None, None, None
    t = title.strip()
    comp = None
    if " - " in t:
        t, comp = t.split(" - ", 1)
    parts = re.split(r"\s+vs?\s+", t, flags=re.IGNORECASE)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip(), comp or "Premier League"
    return None, None, comp

def _rows_from_calendar_text(text: str, source=""):
    cal = Calendar(text)
    rows = []
    for e in cal.events:
        title = getattr(e, "name", None) or getattr(e, "summary", None)
        home, away, comp = _parse_match_title(title)
        dt = getattr(e, "begin", None)
        if dt is None:
            continue
        dt_utc = dt.datetime.replace(tzinfo=dt.datetime.tzinfo or timezone.utc).astimezone(timezone.utc)
        rows.append({
            "Date": dt_utc.date().isoformat(),
            "KickoffUTC": dt_utc.isoformat().replace("+00:00", "Z"),
            "HomeTeam": home,
            "AwayTeam": away,
            "Venue": getattr(e, "location", None),
            "Competition": comp or "Premier League",
            "Source": source,
            "UID": getattr(e, "uid", None) or ""
        })
    return rows

def fixtures_from_ics_texts(texts_with_source):
    """texts_with_source: list of (text, source_label)"""
    all_rows = []
    for text, src in texts_with_source:
        all_rows.extend(_rows_from_calendar_text(text, source=src))
    df = pd.DataFrame(all_rows)
    if df.empty:
        return pd.DataFrame(columns=["Date","HomeTeam","AwayTeam"])
    if "Competition" in df.columns:
        df = df[df["Competition"].str.contains("Premier League", case=False, na=True)]
    df = df.dropna(subset=["HomeTeam","AwayTeam"])
    if "UID" in df.columns:
        df = df.drop_duplicates(subset=["UID","Date","HomeTeam","AwayTeam"])
    else:
        df = df.drop_duplicates(subset=["Date","HomeTeam","AwayTeam"])
    df = df.sort_values(["Date","HomeTeam","AwayTeam"]).reset_index(drop=True)
    return df[["Date","HomeTeam","AwayTeam"]]
