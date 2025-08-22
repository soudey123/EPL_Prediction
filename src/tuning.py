import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from .model import predict_proba, LABEL2ID

def tune_draw_boost(model, feats_df, frac=0.15, grid=None):
    if grid is None:
        grid = np.arange(1.0, 1.61, 0.05)
    cutoff = feats_df["Date"].quantile(1.0 - frac)
    valid = feats_df[feats_df["Date"] > cutoff].dropna(subset=["label"]).copy()
    if len(valid) < 100:
        return 1.0, None, pd.DataFrame({"b":[1.0], "logloss":[np.nan]})
    P = predict_proba(model, valid)
    y = valid["label"].map(LABEL2ID).astype(int).values
    scores = []
    for b in grid:
        Q = P.copy()
        Q[:,1] *= b
        Q = Q / Q.sum(axis=1, keepdims=True)
        ll = log_loss(y, Q, labels=[0,1,2])
        scores.append((float(b), float(ll)))
    scores_df = pd.DataFrame(scores, columns=["b","logloss"]).sort_values("logloss")
    best_b, best_ll = scores_df.iloc[0]["b"], scores_df.iloc[0]["logloss"]
    return float(best_b), float(best_ll), scores_df

def solve_boost_for_target_mean_draw(P, target=0.25, lo=0.5, hi=3.0, tol=1e-4, max_iter=50):
    def mean_draw(b):
        Q = P.copy()
        Q[:,1] *= b
        Q = Q / Q.sum(axis=1, keepdims=True)
        return float(Q[:,1].mean())
    m_lo = mean_draw(lo)
    m_hi = mean_draw(hi)
    m_1 = mean_draw(1.0)
    if abs(m_1 - target) <= tol:
        return 1.0, m_1
    if not (min(m_lo, m_hi) <= target <= max(m_lo, m_hi)):
        return (lo if abs(m_lo - target) < abs(m_hi - target) else hi), (m_lo if abs(m_lo - target) < abs(m_hi - target) else m_hi)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        m_mid = mean_draw(mid)
        if abs(m_mid - target) <= tol:
            return mid, m_mid
        if (m_lo - target) * (m_mid - target) <= 0:
            hi, m_hi = mid, m_mid
        else:
            lo, m_lo = mid, m_mid
    return mid, m_mid
