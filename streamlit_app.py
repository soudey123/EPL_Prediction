import os
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from src.ingest import get_historic_premier_data, load_manual_csv
from src.features import build_features
from src.model import train_models, predict_proba, TARGETS
from src.predict import fetch_fixtures_api, prepare_prediction_rows
from src.utils import brier_score

st.set_page_config(page_title="EPL Predictor 2025–26", layout="wide")

st.title("🏟️ EPL Predictor — Season 2025–26")
st.caption("End-to-end ML tool to forecast Premier League match results (H/D/A).")

with st.expander("ℹ️ How it works"):
    st.markdown("""
    **Pipeline**  
    1) Ingest historical results (Football-Data) and optional First Division files you provide.  
    2) Engineer features: rolling form, goals, head-to-head aggregates, Elo rating & win-prob proxy.  
    3) Train two models (LogReg & XGBoost) with time-aware CV and probability calibration.  
    4) Pull **2025–26 fixtures** via API-Football (or upload fixtures CSV) and generate predictions.  
    5) Export predictions and view feature importance.
    """)

st.header("1) Historical Data")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Fetch Football-Data (1993 → last season)")
    if st.button("Download & Build Dataset"):
        with st.spinner("Downloading Premier League CSVs (Football-Data)..."):
            try:
                hist = get_historic_premier_data(start=1993)
                st.session_state["historic"] = hist
                st.success(f"Loaded {len(hist):,} matches from {hist['SeasonStart'].min()}–{hist['SeasonStart'].max()+1}")
            except Exception as e:
                st.error(f"Download failed: {e}")

with col2:
    st.subheader("Or upload your historic CSV")
    up = st.file_uploader("CSV with at least: Date, HomeTeam, AwayTeam, FTHG, FTAG", type=["csv"])
    if up is not None:
        df = pd.read_csv(up)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        st.session_state["historic"] = df
        st.success(f"Uploaded {len(df):,} rows")

if "historic" not in st.session_state:
    st.info("Load or upload historic data to proceed.")
    st.stop()

st.header("2) Feature Engineering & Training")
if st.button("Build features & train models"):
    with st.spinner("Engineering features and training models..."):
        feats = build_features(st.session_state["historic"])
        model, info = train_models(feats)
        st.session_state["features"] = feats
        st.session_state["model"] = model
        st.session_state["model_info"] = info
        st.success("Models trained and calibrated.")
        st.write("Cross-validated log loss:", info["cv_logloss"])

if "model" not in st.session_state:
    st.info("Train models to continue.")
    st.stop()

st.header("3) Fixtures for 2025–26 & Predictions")
mode = st.radio("How to get fixtures?", ["API-Football (recommended)", "Upload CSV"])

if mode == "API-Football (recommended)":
    if st.button("Fetch fixtures via API-Football"):
        try:
            fx = fetch_fixtures_api(season=2025, league=39)
            st.session_state["fixtures"] = fx
            st.success(f"Loaded {len(fx):,} fixtures for 2025–26.")
        except Exception as e:
            st.error(str(e))
else:
    up2 = st.file_uploader("Upload fixtures CSV (Date, HomeTeam, AwayTeam)", type=["csv"], key="fx")
    if up2 is not None:
        fx = pd.read_csv(up2)
        fx["Date"] = pd.to_datetime(fx["Date"], errors="coerce")
        st.session_state["fixtures"] = fx
        st.success(f"Uploaded {len(fx):,} fixtures.")

if "fixtures" not in st.session_state:
    st.info("Load fixtures to make predictions.")
    st.stop()

if st.button("Predict all fixtures"):
    with st.spinner("Preparing rows & predicting..."):
        hist_feats, future_rows = prepare_prediction_rows(st.session_state["historic"], st.session_state["fixtures"])
        probs = predict_proba(st.session_state["model"], future_rows)
        pred_df = future_rows[["Date","HomeTeam","AwayTeam"]].copy()
        pred_df["p_H"] = probs[:,0]
        pred_df["p_D"] = probs[:,1]
        pred_df["p_A"] = probs[:,2]
        pred_df["Pick"] = pred_df[["p_H","p_D","p_A"]].idxmax(axis=1).str.replace("p_","")
        st.session_state["predictions"] = pred_df
        st.success("Predictions ready.")

if "predictions" in st.session_state:
    st.subheader("Predicted probabilities — 2025–26")
    st.dataframe(st.session_state["predictions"].sort_values("Date").reset_index(drop=True))
    st.download_button("⬇️ Download CSV", data=st.session_state["predictions"].to_csv(index=False), file_name="epl_2025_26_predictions.csv")

st.header("4) Model Explainability & Evaluation")
if "features" in st.session_state:
    feats = st.session_state["features"]
    if "label" in feats.columns:
        # quick backtest score on last season segment
        cutoff = feats["Date"].quantile(0.85)
        test = feats[feats["Date"]>cutoff]
        if len(test) > 50:
            probs = predict_proba(st.session_state["model"], test)
            y = test["label"].values
            from sklearn.metrics import log_loss, accuracy_score
            ll = log_loss(y, probs, labels=["H","D","A"])
            brier = float(np.mean(np.sum((np.eye(3)[pd.Series(y).map({'H':0,'D':1,'A':2}).values] - probs)**2, axis=1)))
            st.write(f"Backtest Log Loss: **{ll:.3f}**")
            st.write(f"Brier Score: **{brier:.3f}**")

    # Feature importances (XGBoost if selected as best; otherwise coefficients proxy)
    info = st.session_state.get("model_info", {})
    feat_names = info.get("features", [])
    st.write("Top feature signals (proxy):")
    try:
        # Try to access underlying estimator if XGB
        est = st.session_state["model"].base_estimator.named_steps.get("clf", None)
        if est and hasattr(est, "feature_importances_"):
            fi = pd.Series(est.feature_importances_, index=feat_names).sort_values(ascending=False).head(20)
            st.bar_chart(fi)
        else:
            st.info("Feature importances not available (Logistic selected).")
    except Exception:
        st.info("Feature importance unavailable.")
