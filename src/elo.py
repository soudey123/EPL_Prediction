import pandas as pd
import numpy as np

class Elo:
    def __init__(self, k=20, base=1500, home_adv=60):
        self.k = k
        self.base = base
        self.home_adv = home_adv
        self.ratings = {}  # team -> rating

    def get(self, team):
        return self.ratings.get(team, self.base)

    def expected_result(self, ra, rb, is_home=True):
        # Elo win probability for A vs B; add home advantage to A
        if is_home: ra = ra + self.home_adv
        return 1 / (1 + 10 ** ((rb - ra)/400))

    def update(self, home, away, outcome):
        # outcome: 1 = home win, 0.5 = draw, 0 = away win (home perspective)
        ra = self.get(home)
        rb = self.get(away)
        ea = self.expected_result(ra, rb, True)
        eb = 1 - ea
        self.ratings[home] = ra + self.k * (outcome - ea)
        self.ratings[away] = rb + self.k * ((1 - outcome) - eb)

def add_elo_features(df):
    """
    df: chronological matches with columns [Date, HomeTeam, AwayTeam, FTHG, FTAG]
    Returns df with Elo pre-match ratings and win prob feature.
    """
    df = df.sort_values("Date").copy()
    elo = Elo()
    elo_home = []
    elo_away = []
    elo_home_wp = []

    for _, row in df.iterrows():
        h, a = row["HomeTeam"], row["AwayTeam"]
        ra, rb = elo.get(h), elo.get(a)
        elo_home.append(ra)
        elo_away.append(rb)
        elo_home_wp.append(elo.expected_result(ra, rb, True))
        # update after result
        hg, ag = row.get("FTHG", np.nan), row.get("FTAG", np.nan)
        if pd.notna(hg) and pd.notna(ag):
            if hg > ag: outcome = 1.0
            elif hg < ag: outcome = 0.0
            else: outcome = 0.5
            elo.update(h, a, outcome)

    df["elo_home"] = elo_home
    df["elo_away"] = elo_away
    df["elo_home_wp"] = elo_home_wp
    return df
