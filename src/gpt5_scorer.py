import os, json, hashlib
import numpy as np

try:
    from openai import OpenAI
    _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception:
    _client = None

CACHE_DIR = os.path.join(os.getcwd(), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_PATH = os.path.join(CACHE_DIR, "gpt5_probs.jsonl")

SYSTEM = (
    "You are a calibrated football forecaster. "
    "Given a compact pre-match numeric card, return a strict JSON object with keys p_H, p_D, p_A "
    "that sum to 1. Use a realistic draw rate when teams are close."
)
PROMPT_PREFIX = "Card: "

def _cache_get(key: str):
    if not os.path.exists(CACHE_PATH):
        return None
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                if obj.get("key") == key:
                    return obj.get("data")
            except Exception:
                continue
    return None

def _cache_put(key: str, data: dict):
    with open(CACHE_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, "data": data}) + "\n")

def feature_card(row: dict) -> dict:
    elo_home = float(row.get("elo_home", 1500.0))
    elo_away = float(row.get("elo_away", 1500.0))
    return {
        "home": row.get("HomeTeam"),
        "away": row.get("AwayTeam"),
        "elo_home": elo_home,
        "elo_away": elo_away,
        "elo_gap": float(elo_home - elo_away),
        "h_form_gd_r5": float(row.get("h_gd_r5", 0.0)),
        "a_form_gd_r5": float(row.get("a_gd_r5", 0.0)),
        "h_win_rate_r5": float(row.get("h_win_r5", 0.0)),
        "a_win_rate_r5": float(row.get("a_win_r5", 0.0)),
        "h_draw_rate_r10": float(row.get("h_draw_r10", 0.0)),
        "a_draw_rate_r10": float(row.get("a_draw_r10", 0.0)),
        "home_advantage": 1
    }

def gpt5_probs_from_card(card: dict) -> np.ndarray:
    try:
        key = hashlib.md5(json.dumps(card, sort_keys=True).encode()).hexdigest()
        cached = _cache_get(key)
        if cached:
            p = np.array([cached["p_H"], cached["p_D"], cached["p_A"]], dtype=float)
            p = np.clip(p, 1e-6, 1.0); p = p / p.sum()
            return p
        if _client is None:
            p = np.array([0.40, 0.28, 0.32], dtype=float)
            return p / p.sum()
        resp = _client.chat.completions.create(
            model="gpt-5",
            temperature=0.0,
            response_format={"type":"json_object"},
            messages=[
                {"role":"system", "content": SYSTEM},
                {"role":"user", "content": PROMPT_PREFIX + json.dumps(card)}
            ],
            max_tokens=80,
        )
        data = json.loads(resp.choices[0].message.content)
        p = np.array([float(data["p_H"]), float(data["p_D"]), float(data["p_A"])], dtype=float)
        p = np.clip(p, 1e-6, 1.0); p = p / p.sum()
        _cache_put(key, {"p_H": float(p[0]), "p_D": float(p[1]), "p_A": float(p[2])})
        return p
    except Exception:
        p = np.array([0.38, 0.27, 0.35], dtype=float)
        return p / p.sum()
