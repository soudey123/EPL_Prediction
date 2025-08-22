import numpy as np, pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss
from .model import predict_proba, LABEL2ID
from .gpt5_scorer import feature_card, gpt5_probs_from_card

def _gpt_block(df_rows: pd.DataFrame) -> np.ndarray:
    out = np.zeros((len(df_rows),3))
    for i, (_, r) in enumerate(df_rows.iterrows()):
        card = feature_card(r)
        out[i,:] = gpt5_probs_from_card(card)
    return out

def _reorder_meta(classes_, proba):
    idx = {c:i for i,c in enumerate(classes_)}
    cols = [idx[0], idx[1], idx[2]]
    return proba[:, cols]

def train_stacked(base_model, feats_df: pd.DataFrame, seasons_back: int = 5):
    max_year = feats_df["Date"].dt.year.max()
    cutoff_year = max_year - seasons_back
    df = feats_df[(feats_df["Date"].dt.year >= cutoff_year) & (feats_df["label"].notna())].copy()
    if len(df) < 300:
        raise RuntimeError("Not enough data to train stacked model. Lower seasons_back or add data.")

    y_all = df["label"].map(LABEL2ID).astype(int).values
    tscv = TimeSeriesSplit(n_splits=4)
    oof_metaX = np.zeros((len(df), 6))
    oof_y = np.zeros(len(df), dtype=int)

    for tr_idx, te_idx in tscv.split(df):
        te_rows = df.iloc[te_idx]
        ml_p = predict_proba(base_model, te_rows)
        gpt_p = _gpt_block(te_rows)
        oof_metaX[te_idx,:] = np.hstack([ml_p, gpt_p])
        oof_y[te_idx] = y_all[te_idx]

    meta = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=400, multi_class="multinomial"))
    ])
    meta.fit(oof_metaX, oof_y)
    cv_ll = log_loss(oof_y, _reorder_meta(meta.classes_, meta.predict_proba(oof_metaX)), labels=[0,1,2])
    return meta, {"cv_logloss_meta": float(cv_ll), "samples": int(len(df))}

def stacked_predict(base_model, meta_model, future_rows: pd.DataFrame) -> np.ndarray:
    ml_p = predict_proba(base_model, future_rows)
    gpt_p = _gpt_block(future_rows)
    metaX = np.hstack([ml_p, gpt_p])
    proba = meta_model.predict_proba(metaX)
    return _reorder_meta(meta_model.classes_, proba)
