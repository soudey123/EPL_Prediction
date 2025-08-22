import os, io, requests
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="EPL Predictor 2025–26 (ML + GPT‑5 + Draw Tuning)", layout="wide")
st.title("🏟️ EPL Predictor — 2025–26 (ML + GPT‑5 + Draw Tuning)")

from src.ingest import get_historic_premier_data
from src.features import build_features
from src.model import train_models, predict_proba, LABEL2ID
from src.predict import fetch_fixtures_api, prepare_prediction_rows

try:
    from src.ics_tools import fixtures_from_ics_texts
    ICS_AVAILABLE = True
except Exception:
    ICS_AVAILABLE = False

from src.stacking import train_stacked, stacked_predict
from src.gpt5_scorer import gpt5_probs_from_card, feature_card
from src.tuning import tune_draw_boost, solve_boost_for_target_mean_draw

with st.expander("ℹ️ What’s here"):
    st.markdown("""
    - **Prediction engine:** ML‑only, GPT‑5‑only, Blend, or Stacked meta‑model.  
    - **Draw boost:** auto‑tune via log loss (last 15%) or solve for a target mean draw rate.  
    """)

st.header("1) Historical Data")
c1, c2 = st.columns(2)
with c1:
    if st.button("Download & Build Dataset (Football-Data)"):
        with st.spinner("Downloading and building…"):
            try:
                hist = get_historic_premier_data(start=1993)
                st.session_state["historic"] = hist
                st.success(f"Loaded {len(hist):,} matches.")
            except Exception as e:
                st.error(f"Download failed: {e}")
with c2:
    up = st.file_uploader("Or upload historic CSV (Date, HomeTeam, AwayTeam, FTHG, FTAG)", type=["csv"])
    if up is not None:
        df = pd.read_csv(up)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        st.session_state["historic"] = df
        st.success(f"Uploaded {len(df):,} rows")

if "historic" not in st.session_state:
    st.info("Load or upload historic data to proceed.")
    st.stop()

st.header("2) Features & Base Model")
if st.button("Build features & train model"):
    with st.spinner("Engineering features and training…"):
        feats = build_features(st.session_state["historic"])
        model, info = train_models(feats)
        st.session_state["features"] = feats
        st.session_state["base_model"] = model
        st.session_state["base_info"] = info
        st.success("Base model trained.")
        st.write("Base CV log loss:", info["cv_logloss"])

if "base_model" not in st.session_state:
    st.info("Train base model to continue.")
    st.stop()

st.header("3) Fixtures (2025–26)")
choices = ["API-Football", "Upload CSV"]
if ICS_AVAILABLE:
    choices.append("ICS (URL or upload)")
mode = st.radio("Choose source:", choices)

if mode == "API-Football":
    if st.button("Fetch via API-Football"):
        try:
            fx = fetch_fixtures_api(season=2025, league=39)
            st.session_state["fixtures"] = fx
            st.success(f"Loaded {len(fx):,} fixtures.")
        except Exception as e:
            st.error(str(e))
elif mode == "Upload CSV":
    up2 = st.file_uploader("Upload fixtures CSV (Date, HomeTeam, AwayTeam)", type=["csv"], key="fx")
    if up2 is not None:
        fx = pd.read_csv(up2)
        fx["Date"] = pd.to_datetime(fx["Date"], errors="coerce")
        st.session_state["fixtures"] = fx
        st.success(f"Uploaded {len(fx):,} fixtures.")
else:
    st.write("Paste ICS URLs (one per line) and/or upload .ics files, then click Build.")
    urls_text = st.text_area("ICS URLs", height=110)
    ics_files = st.file_uploader("Upload .ics files", type=["ics"], accept_multiple_files=True, key="ics")
    if st.button("Build fixtures from ICS"):
        texts = []
        for u in (urls_text or "").splitlines():
            u = u.strip()
            if not u: continue
            try:
                r = requests.get(u, timeout=30); r.raise_for_status()
                texts.append((r.text, u))
            except Exception as e:
                st.warning(f"Failed to fetch {u}: {e}")
        for f in ics_files or []:
            try:
                texts.append((f.getvalue().decode("utf-8", errors="ignore"), f.name))
            except Exception as e:
                st.warning(f"Read failed for {getattr(f,'name','(file)')}: {e}")
        if not texts:
            st.error("No ICS input provided.")
        else:
            from src.ics_tools import fixtures_from_ics_texts as parse_ics
            try:
                df_fx = parse_ics(texts)
                if df_fx.empty: st.error("Parsed 0 fixtures from ICS.")
                else:
                    df_fx["Date"] = pd.to_datetime(df_fx["Date"], errors="coerce")
                    st.session_state["fixtures"] = df_fx
                    st.success(f"Built {len(df_fx):,} fixtures from ICS.")
                    st.dataframe(df_fx.head(20))
                    st.download_button("⬇️ Download fixtures CSV", data=df_fx.to_csv(index=False), file_name="fixtures_2025_26.csv")
            except Exception as e:
                st.error(f"ICS parse error: {e}")

if "fixtures" not in st.session_state:
    st.info("Load fixtures to continue.")
    st.stop()

st.header("4) Prediction Engine")
engine = st.radio("Engine:", ["ML only", "GPT‑5 only", "Blend (α·ML + (1−α)·GPT‑5)", "Stacked (meta‑model)"])
if engine == "Blend (α·ML + (1−α)·GPT‑5)":
    alpha = st.slider("Blending weight α (ML share)", 0.0, 1.0, 0.75, 0.05)
if engine == "Stacked (meta‑model)":
    seasons_back = st.slider("Seasons back (stacking window)", 3, 8, 5, 1)
    if st.button("Train stacked meta‑model"):
        with st.spinner("Training stacked meta‑model…"):
            try:
                meta_model, meta_info = train_stacked(st.session_state["base_model"], st.session_state["features"], seasons_back=seasons_back)
                st.session_state["meta_model"] = meta_model
                st.success(f"Stacked meta-model trained. CV log loss: {meta_info['cv_logloss_meta']:.3f} on {meta_info['samples']} samples")
            except Exception as e:
                st.error(f"Stacking failed: {e}")

st.header("5) Draw Boost")
colL, colR = st.columns(2)
with colL:
    if st.button("Auto‑tune (log loss, last 15%)"):
        b_star, ll, table = tune_draw_boost(st.session_state["base_model"], st.session_state["features"], frac=0.15)
        st.session_state["draw_boost_b"] = b_star
        st.success(f"Recommended b = {b_star:.2f}  (val log loss {ll:.3f})")
        st.dataframe(table.head(10))
with colR:
    target = st.number_input("Target mean draw rate", min_value=0.10, max_value=0.40, value=0.25, step=0.01)
manual_b = st.slider("Manual Draw Boost × (fallback if no auto‑tune/target)", 0.5, 1.6, 1.25, 0.05)

st.header("6) Predict Season")
if st.button("Predict all fixtures"):
    with st.spinner("Preparing rows & predicting…"):
        hist_feats, future_rows = prepare_prediction_rows(st.session_state["historic"], st.session_state["fixtures"])

        if engine == "ML only":
            probs = predict_proba(st.session_state["base_model"], future_rows)
        elif engine == "GPT‑5 only":
            P_gpt = np.zeros((len(future_rows),3))
            for i, (_, row) in enumerate(future_rows.iterrows()):
                P_gpt[i,:] = gpt5_probs_from_card(feature_card(row))
            probs = P_gpt
        elif engine == "Blend (α·ML + (1−α)·GPT‑5)":
            P_ml = predict_proba(st.session_state["base_model"], future_rows)
            P_gpt = np.zeros_like(P_ml)
            for i, (_, row) in enumerate(future_rows.iterrows()):
                P_gpt[i,:] = gpt5_probs_from_card(feature_card(row))
            probs = alpha * P_ml + (1.0 - alpha) * P_gpt
        else:
            if "meta_model" not in st.session_state:
                st.error("Train the stacked meta‑model first.")
                st.stop()
            probs = stacked_predict(st.session_state["base_model"], st.session_state["meta_model"], future_rows)

        # Draw boost
        b = st.session_state.get("draw_boost_b", None)
        if b is None and target:
            b, achieved = solve_boost_for_target_mean_draw(probs, target=target)
            st.info(f"Solved b = {b:.2f} for target mean draw {target:.2f}")
        elif b is None:
            b = manual_b
        probs[:,1] *= b
        probs = probs / probs.sum(axis=1, keepdims=True)

        pred_df = future_rows[["Date","HomeTeam","AwayTeam"]].copy()
        pred_df["p_H"] = probs[:,0]
        pred_df["p_D"] = probs[:,1]
        pred_df["p_A"] = probs[:,2]
        pred_df["Pick"] = pred_df[["p_H","p_D","p_A"]].idxmax(axis=1).str.replace("p_","")
        st.session_state["predictions"] = pred_df
        st.success("Predictions ready.")

if "predictions" in st.session_state:
    st.subheader("Season predictions")
    P = st.session_state["predictions"][["p_H","p_D","p_A"]].values
    st.write(f"Mean probs — H: {P[:,0].mean():.3f}, D: {P[:,1].mean():.3f}, A: {P[:,2].mean():.3f}")
    st.dataframe(st.session_state["predictions"].sort_values("Date").reset_index(drop=True))
    st.download_button("⬇️ Download CSV", data=st.session_state["predictions"].to_csv(index=False), file_name="epl_2025_26_predictions_calibrated.csv")
