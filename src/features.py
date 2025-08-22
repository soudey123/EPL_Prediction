import pandas as pd
import numpy as np
from .elo import add_elo_features
from .utils import result_label

def rolling_team_stats(matches, windows=(5,10)):
    """
    Build per-team rolling form features prior to each match.
    matches: chronological DataFrame with [Date, HomeTeam, AwayTeam, FTHG, FTAG]
    """
    df = matches.sort_values("Date").copy()

    # Long format (one row per team per match)
    long_rows = []
    for _, r in df.iterrows():
        long_rows.append({"Date": r["Date"], "Team": r["HomeTeam"], "Opp": r["AwayTeam"],
                          "is_home":1, "gf":r["FTHG"], "ga":r["FTAG"]})
        long_rows.append({"Date": r["Date"], "Team": r["AwayTeam"], "Opp": r["HomeTeam"],
                          "is_home":0, "gf":r["FTAG"], "ga":r["FTHG"]})
    long = pd.DataFrame(long_rows)

    # Sort by team/date and make a stable RowID to align rolling outputs
    long = long.sort_values(["Team","Date"]).reset_index(drop=True)
    long["gd"] = long["gf"] - long["ga"]
    long["win"] = (long["gd"]>0).astype(int)
    long["draw"] = (long["gd"]==0).astype(int)
    long["loss"] = (long["gd"]<0).astype(int)
    long["RowID"] = long.index

    base_cols = ["gf","ga","gd","win","draw","loss"]
    # Compute rolling means per team using RowID index; shift(1) so stats are prior to current match.
    for w in windows:
        rolled = (
            long.groupby("Team")
                .apply(lambda g: g.set_index("RowID")[base_cols]
                                  .rolling(window=w, min_periods=1)
                                  .mean()
                                  .shift(1))
                .reset_index()
        )
        for c in base_cols:
            long = long.merge(
                rolled[["Team","RowID",c]].rename(columns={c: f"{c}_r{w}"}),
                on=["Team","RowID"],
                how="left"
            )

    # Bring back to match-level (home vs away features) using the last row for that (Team, Date)
    out_rows = []
    df_sorted = df.sort_values("Date")
    for _, m in df_sorted.iterrows():
        # pick the latest pre-match row for each team at this date
        home_rows = long[(long["Team"]==m["HomeTeam"]) & (long["Date"]<=m["Date"])]
        away_rows = long[(long["Team"]==m["AwayTeam"]) & (long["Date"]<=m["Date"])]
        if len(home_rows)==0 or len(away_rows)==0:
            continue
        hr = home_rows.iloc[-1]; ar = away_rows.iloc[-1]
        row = {"Date":m["Date"],"HomeTeam":m["HomeTeam"],"AwayTeam":m["AwayTeam"],
               "FTHG":m.get("FTHG",np.nan),"FTAG":m.get("FTAG",np.nan),"FTR":m.get("FTR",np.nan)}
        # prefix features (only *_rW columns)
        for c in long.columns:
            if any([c.endswith(f"_r{w}") for w in windows]):
                row[f"h_{c}"] = hr[c]
                row[f"a_{c}"] = ar[c]
        out_rows.append(row)

    out = pd.DataFrame(out_rows).sort_values("Date")

    # add elo features AFTER rolling so Elo uses chronological match list
    out = add_elo_features(out)

    # label
    if "FTHG" in out.columns and "FTAG" in out.columns:
        out["label"] = out.apply(lambda r: result_label(r["FTHG"], r["FTAG"]), axis=1)
    return out

def build_features(historic_matches):
    feats = rolling_team_stats(historic_matches)
    # minimal NA handling
    feat_cols = [c for c in feats.columns if c not in ["Date","HomeTeam","AwayTeam","FTHG","FTAG","FTR","label"]]
    feats[feat_cols] = feats[feat_cols].fillna(feats[feat_cols].median())
    return feats
