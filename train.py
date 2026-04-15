"""
train.py
--------
Downloads historical AAPL data (2018-2024), engineers features,
trains Linear Regression (baseline) and Random Forest (advanced)
regressors on an 80/20 time-ordered split, evaluates both with
RMSE and MAE, selects the best model, and persists it plus its
scaler and metadata to models/.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import joblib
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

MODELS_DIR = Path(__file__).parent / "models"


# ── feature engineering ──────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature matrix from an OHLCV DataFrame.

    Features (35 total)
    -------------------
    Close_lag_1  … Close_lag_30   – 30 lagged closing prices
    MA7, MA14, MA30              – simple moving averages
    STD7                         – 7-day rolling std of Close
    Trend                        – 5-day rolling mean of daily pct change

    Target
    ------
    Target = next trading day's Close  (Close shifted by −1)
    """
    out = df[["Close"]].copy()

    # ── 30 lag prices ──
    for i in range(1, 31):
        out[f"Close_lag_{i}"] = out["Close"].shift(i)

    # ── moving averages ──
    out["MA7"]  = out["Close"].rolling(window=7).mean()
    out["MA14"] = out["Close"].rolling(window=14).mean()
    out["MA30"] = out["Close"].rolling(window=30).mean()

    # ── volatility ──
    out["STD7"] = out["Close"].rolling(window=7).std()

    # ── trend ── captures short-term price momentum
    out["Price_Change"] = out["Close"].pct_change()
    out["Trend"] = out["Price_Change"].rolling(window=5).mean()

    # ── RSI (Relative Strength Index) ──
    delta = out["Close"].diff()
    gain = delta.where(delta > 0, 0).ewm(com=13, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(com=13, adjust=False).mean()
    rs = gain / loss
    out["RSI"] = 100 - (100 / (1 + rs))

    # ── MACD (Moving Average Convergence Divergence) ──
    ema12 = out["Close"].ewm(span=12, adjust=False).mean()
    ema26 = out["Close"].ewm(span=26, adjust=False).mean()
    out["MACD"] = ema12 - ema26
    out["MACD_Signal"] = out["MACD"].ewm(span=9, adjust=False).mean()

    # ── target: next day's close ──
    out["Target"] = out["Close"].shift(-1)

    # drop rows that have any NaN (from lags / rolling / target)
    out = out.dropna()

    return out


# ── training pipeline ────────────────────────────────────────────────

def train() -> None:
    # ① download data
    print("[DOWNLOAD] Downloading AAPL data (2018-2024)...")
    raw = yf.download("AAPL", start="2018-01-01", end="2024-12-31", progress=False)

    if raw.empty:
        raise RuntimeError("yfinance returned no data for AAPL")

    # flatten MultiIndex columns that newer yfinance versions produce
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    print(f"    {len(raw)} rows downloaded")

    # ② engineer features
    print("[FEATURES] Engineering features...")
    featured = engineer_features(raw)

    feature_cols = (
        [f"Close_lag_{i}" for i in range(1, 31)]
        + ["MA7", "MA14", "MA30", "STD7", "Trend", "RSI", "MACD", "MACD_Signal"]
    )
    X = featured[feature_cols]
    y = featured["Target"]

    # ③ 80 / 20 time-ordered split (no shuffle – respects chronology)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"[SPLIT] Train: {len(X_train)} samples  |  Test: {len(X_test)} samples")

    # ④ scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # ⑤ train candidates — baseline (Linear Regression) + advanced (Random Forest)
    candidates = {
        "LinearRegression": LinearRegression(),                   # ← baseline model
        "RandomForest": RandomForestRegressor(                    # ← new advanced model
            n_estimators=200,
            max_depth=15,
            random_state=42,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            random_state=42
        ),
    }

    results: dict[str, dict] = {}

    for name, model in candidates.items():
        print(f"[TRAIN] Training {name}...")
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        # ── evaluate with both RMSE and MAE ──
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        mae  = float(mean_absolute_error(y_test, preds))
        results[name] = {"model": model, "rmse": rmse, "mae": mae}
        print(f"    RMSE : {rmse:.4f}")
        print(f"    MAE  : {mae:.4f}")

    # ⑥ model comparison — select the model with the lowest RMSE
    print("\n" + "=" * 60)
    print("  MODEL COMPARISON RESULTS")
    print("=" * 60)
    for name, r in results.items():
        print(f"  {name:25s}  RMSE = {r['rmse']:.4f}   MAE = {r['mae']:.4f}")
    print("-" * 60)

    best_name  = min(results, key=lambda k: results[k]["rmse"])
    best_model = results[best_name]["model"]
    best_rmse  = results[best_name]["rmse"]
    best_mae   = results[best_name]["mae"]

    print(f"  >> Best Model : {best_name}")
    print(f"     RMSE       : {best_rmse:.4f}")
    print(f"     MAE        : {best_mae:.4f}")
    print("=" * 60)

    # ⑦ persist artifacts
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, MODELS_DIR / "model.joblib")
    joblib.dump(scaler,     MODELS_DIR / "scaler.joblib")

    meta = {
        "model_name":    best_name,
        "rmse":          best_rmse,
        "mae":           best_mae,
        "feature_names": feature_cols,
        "ticker":        "AAPL",
        "train_start":   "2018-01-01",
        "train_end":     "2024-12-31",
        "train_samples": len(X_train),
        "test_samples":  len(X_test),
        # full comparison so the API can expose both models' scores
        "model_comparison": {
            name: {"rmse": r["rmse"], "mae": r["mae"]}
            for name, r in results.items()
        },
    }
    joblib.dump(meta, MODELS_DIR / "meta.joblib")

    print(f"[SAVE] Saved to {MODELS_DIR}/  ->  model.joblib, scaler.joblib, meta.joblib")
    print("[DONE] Training complete.")


if __name__ == "__main__":
    train()
