"""
predictor.py
------------
Loads a trained scikit-learn model, scaler, and metadata from models/,
downloads recent price data for a given ticker, builds features that
match train.py exactly, and predicts the next 7 trading days using a
rolling-forward approach.
"""

from datetime import datetime, timedelta
from pathlib import Path

import joblib
import pandas as pd
import yfinance as yf

MODELS_DIR = Path(__file__).parent / "models"


# ── helpers ──────────────────────────────────────────────────────────

def _load_artifacts():
    """Load the trained model, fitted scaler, and training metadata."""
    model  = joblib.load(MODELS_DIR / "model.joblib")
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    meta   = joblib.load(MODELS_DIR / "meta.joblib")
    return model, scaler, meta


def _calculate_confidence(rmse: float, current_price: float, recent_std: float) -> float:
    """
    Calculate confidence based on historical RMSE and the current volatility.
    """
    if current_price <= 0:
        return 0.0
    # Use the max of historical RMSE and current standard deviation as the error estimate
    expected_error = max(rmse, recent_std)
    error_margin = expected_error / current_price
    
    # Scale: an error margin > 5% drops confidence heavily.
    confidence = max(0.0, 1.0 - (error_margin * 10))
    return round(confidence, 2)


def _calculate_risk(recent_std: float, recent_trend: float, current_price: float) -> str:
    """
    Determine risk level by combining relative volatility and recent trend strength.
    """
    if current_price <= 0: return "HIGH"
    
    relative_vol = recent_std / current_price
    
    # High volatility (> 2%) AND unstable trend (negative trend) -> HIGH
    if relative_vol > 0.02 and recent_trend < 0:
        return "HIGH"
    elif relative_vol > 0.01 or recent_trend < 0:
        return "MEDIUM"
    else:
        return "LOW"


def _determine_decision(signal: str, confidence: float, risk_level: str) -> tuple[str, str]:
    """
    Determine final decision and reasoning by factoring in confidence and risk.
    """
    if confidence < 0.5:
        return "HOLD", "Prediction uncertain due to low confidence"
    if risk_level == "HIGH":
        return "HOLD", "High market volatility detected, holding recommended"
    
    if signal == "BUY":
        return signal, "Upward trend detected based on recent data"
    else:
        return signal, "Downward trend detected based on recent data"


# ── feature engineering (must match train.py exactly) ────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the feature matrix from an OHLCV DataFrame.

    Features (35 total) — identical to train.engineer_features()
    -----------------------------------------------------------
    Close_lag_1  … Close_lag_30   – 30 lagged closing prices
    MA7, MA14, MA30              – simple moving averages
    STD7                         – 7-day rolling std of Close
    Trend                        – 5-day rolling mean of daily pct change

    The "Close" and "Target" columns are NOT included in the output;
    only the feature columns are returned.
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

    # ── columns to drop ──the 35 feature columns (must match train.engineer_features)
    feature_cols = (
        [f"Close_lag_{i}" for i in range(1, 31)]
        + ["MA7", "MA14", "MA30", "STD7", "Trend", "RSI", "MACD", "MACD_Signal"]
    )
    return out[feature_cols]


# ── public API ───────────────────────────────────────────────────────

def predict(ticker: str) -> dict:
    """
    Predict the next 7 trading-day closing prices for *ticker*.

    Returns
    -------
    dict with keys:
        ticker                  – uppercased symbol
        current_price           – last known close
        model_used              – name of the best model
        predictions             – list of {date, predicted_price}
        average_predicted_price
        signal                  – "BUY" if avg predicted > current else "SELL"
        reason                  – explainable reasoning for the generated signal
        confidence              – normalized prediction reliability score (0.0–1.0)
        risk_level              – "HIGH" / "MEDIUM" / "LOW" based on volatility and trend
    """
    model, scaler, meta = _load_artifacts()
    feature_names: list[str] = meta["feature_names"]

    # ── download last 6 months of data ──
    end_dt   = datetime.now()
    start_dt = end_dt - timedelta(days=180)
    raw = yf.download(
        ticker,
        start=start_dt.strftime("%Y-%m-%d"),
        end=end_dt.strftime("%Y-%m-%d"),
        progress=False,
    )

    if raw.empty:
        raise ValueError(f"No price data found for ticker '{ticker}'")

    # yfinance may return MultiIndex columns for a single ticker
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    current_price = float(raw["Close"].iloc[-1])
    last_date     = raw.index[-1]

    # ── confidence score & risk level ── based on volatility + recent trend
    rmse = meta.get("rmse", 0)
    recent_std = float(raw["Close"].rolling(window=7).std().iloc[-1])
    recent_trend = float(raw["Close"].pct_change().rolling(window=5).mean().iloc[-1])
    
    confidence = _calculate_confidence(rmse, current_price, recent_std)
    risk_level = _calculate_risk(recent_std, recent_trend, current_price)

    # ── rolling 7-day prediction ──
    predictions: list[dict] = []
    working_df = raw.copy()

    for _ in range(7):
        # rebuild features on the full (growing) history
        feat_df = build_features(working_df)
        feat_df = feat_df.dropna()

        if feat_df.empty:
            raise ValueError(
                f"Not enough historical data to compute features for '{ticker}'"
            )

        # take the latest row, scale, predict
        latest_row    = feat_df[feature_names].iloc[[-1]]
        latest_scaled = scaler.transform(latest_row)
        pred_price    = float(model.predict(latest_scaled)[0])

        # next trading day (skip weekends)
        next_date = last_date + timedelta(days=1)
        while next_date.weekday() >= 5:
            next_date += timedelta(days=1)

        predictions.append({
            "date": next_date.strftime("%Y-%m-%d"),
            "predicted_price": round(pred_price, 2),
        })

        # append a synthetic row so the next iteration can use it
        synthetic = pd.DataFrame(
            {
                "Open":      [pred_price],
                "High":      [pred_price],
                "Low":       [pred_price],
                "Close":     [pred_price],
                "Adj Close": [pred_price],
                "Volume":    [0],
            },
            index=[next_date],
        )
        working_df = pd.concat([working_df, synthetic])
        last_date  = next_date

    avg_predicted = round(
        sum(p["predicted_price"] for p in predictions) / len(predictions), 2
    )
    
    # ── explainability (NEW) ── detail the reasoning behind the signal decision
    if avg_predicted > current_price:
        signal = "BUY"
    else:
        signal = "SELL"
        
    final_decision, reason = _determine_decision(signal, confidence, risk_level)

    return {
        "ticker": ticker.upper(),
        "current_price": round(current_price, 2),
        "model_used": meta.get("model_name", "unknown"),
        "predictions": predictions,
        "average_predicted_price": avg_predicted,
        "signal": signal,
        "final_decision": final_decision,
        "reason": reason,
        "confidence": confidence,
        "risk_level": risk_level,
        "model_accuracy_note": f"Model error ≈ {round(rmse, 2)} USD based on historical test data",
    }
