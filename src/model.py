import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

TARGETS = ["H","D","A"]
LABEL2ID = {"H":0, "D":1, "A":2}
ID2LABEL = {v:k for k,v in LABEL2ID.items()}

def _encode_y(y_series):
    return y_series.map(LABEL2ID).astype(int)

def _class_weights(y_int):
    counts = np.bincount(y_int, minlength=3).astype(float)
    # inverse-frequency weights normalized to mean 1.0
    inv = 1.0 / np.maximum(counts, 1.0)
    w = inv * (3.0 / inv.sum())
    return {0:w[0], 1:w[1], 2:w[2]}, w

def _reorder_proba(classes_, proba):
    class_to_idx = {c:i for i,c in enumerate(classes_)}
    idxs = [class_to_idx[LABEL2ID[t]] for t in TARGETS]
    return proba[:, idxs]

def train_models(features_df, seed=42):
    df = features_df.dropna(subset=["label"]).copy()
    X = df.drop(columns=["Date","HomeTeam","AwayTeam","FTHG","FTAG","FTR","label"])
    y = _encode_y(df["label"])

    # compute class weights and per-sample weights
    cw_dict, cw_arr = _class_weights(y.values)
    sample_w = np.array([cw_arr[i] for i in y.values])

    # define pipelines
    lr = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", LogisticRegression(max_iter=400, n_jobs=None, multi_class="multinomial",
                                   class_weight=cw_dict))
    ])
    xgb = XGBClassifier(
        n_estimators=700, max_depth=5, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
        objective="multi:softprob", num_class=3, random_state=seed, reg_lambda=1.0,
        min_child_weight=2, gamma=0.0
    )
    xgb_pipe = Pipeline([("clf", xgb)])

    tscv = TimeSeriesSplit(n_splits=5)
    results = []
    for name, pipe in [("LogReg", lr), ("XGBoost", xgb_pipe)]:
        fold_losses = []
        for train_idx, test_idx in tscv.split(X):
            Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
            ytr, yte = y.iloc[train_idx], y.iloc[test_idx]
            sw_tr = sample_w[train_idx]
            # fit with sample weights
            if name == "LogReg":
                pipe.fit(Xtr, ytr, clf__sample_weight=sw_tr)
            else:
                pipe.fit(Xtr, ytr, clf__sample_weight=sw_tr)
            # calibrate on last 10% of train (with weights)
            cut = int(len(Xtr)*0.9)
            cal = CalibratedClassifierCV(pipe, method="isotonic", cv="prefit")
            cal.fit(Xtr.iloc[cut:], ytr.iloc[cut:], sample_weight=sw_tr[cut:])
            p = cal.predict_proba(Xte)
            loss = log_loss(yte, p, labels=[0,1,2])
            fold_losses.append(loss)
        results.append((name, float(np.mean(fold_losses))))

    results.sort(key=lambda x: x[1])
    best_name = results[0][0]
    best_pipe = lr if best_name == "LogReg" else xgb_pipe
    # final fit on all data with weights + calibrate
    if best_name == "LogReg":
        best_pipe.fit(X, y, clf__sample_weight=sample_w)
    else:
        best_pipe.fit(X, y, clf__sample_weight=sample_w)
    cal = CalibratedClassifierCV(best_pipe, method="isotonic", cv=3)
    cal.fit(X, y, sample_weight=sample_w)
    return cal, {"cv_logloss": results, "features": list(X.columns), "class_weights": cw_dict}

def predict_proba(model, feature_rows):
    X = feature_rows.drop(columns=["Date","HomeTeam","AwayTeam","FTHG","FTAG","FTR","label"], errors="ignore")
    p = model.predict_proba(X)
    classes_ = model.classes_
    return _reorder_proba(classes_, p)
