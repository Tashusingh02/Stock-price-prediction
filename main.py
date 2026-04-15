"""
main.py
-------
FastAPI backend for the Stock Price Predictor.

Endpoints
---------
GET /             – health check, confirms model artifacts exist
GET /predict      – predict next 7 trading days for a ticker
GET /model-info   – return training metadata and model comparison results
"""

from pathlib import Path

import joblib
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from predictor import predict as run_prediction

app = FastAPI(
    title="Stock Price Predictor API",
    version="1.0.0",
    description="Predict next-week stock closing prices using a trained ML model.",
)

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR = Path(__file__).parent / "models"
FRONTEND_DIR = Path(__file__).parent / "frontend"

MODEL_PATH  = MODELS_DIR / "model.joblib"
SCALER_PATH = MODELS_DIR / "scaler.joblib"
META_PATH   = MODELS_DIR / "meta.joblib"


# ── helpers ──────────────────────────────────────────────────────────

def _check_model_exists() -> None:
    """Raise 503 if any required artifact is missing."""
    missing = [
        p.name for p in (MODEL_PATH, SCALER_PATH, META_PATH) if not p.exists()
    ]
    if missing:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Model not trained yet. Run train.py first.",
                "missing_files": missing,
            },
        )


# ── endpoints ────────────────────────────────────────────────────────

@app.get("/health", tags=["Health"])
def health_check():
    """
    Health check — confirms the API is running and model files are present.
    """
    model_ready = all(p.exists() for p in (MODEL_PATH, SCALER_PATH, META_PATH))

    return {
        "status": "ok",
        "model_ready": model_ready,
        "message": (
            "Model loaded and ready for predictions."
            if model_ready
            else "Model not found. Run train.py to train the model."
        ),
    }


@app.get("/predict", tags=["Prediction"])
def predict_endpoint(
    ticker: str = Query(
        ...,
        min_length=1,
        max_length=10,
        description="Stock ticker symbol, e.g. AAPL",
        examples=["AAPL", "GOOG", "MSFT"],
    ),
):
    """
    Predict the next 7 trading-day closing prices for **ticker**.

    Returns current price, list of predictions with dates,
    the average predicted price, and a BUY / SELL signal.
    """
    _check_model_exists()

    # normalise to uppercase, strip whitespace
    ticker = ticker.strip().upper()
    if not all(c.isalnum() or c in ".-" for c in ticker):
        raise HTTPException(
            status_code=400,
            detail={"error": "Invalid ticker symbol. Use formats like AAPL, MSFT, GOOG"},
        )

    try:
        result = run_prediction(ticker)
    except ValueError as exc:
        # bad ticker / no data / not enough history
        raise HTTPException(
            status_code=400,
            detail={"error": str(exc)},
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={"error": f"Unexpected error: {exc}"},
        )

    return result


@app.get("/model-info", tags=["Model"])
def model_info():
    """
    Return training metadata including:
    - **best_model**: name, RMSE, and MAE of the selected model
    - **model_comparison**: RMSE and MAE for every candidate trained
    - **training_details**: ticker, date range, sample counts, feature list
    """
    _check_model_exists()

    try:
        meta = joblib.load(META_PATH)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={"error": f"Failed to load model metadata: {exc}"},
        )

    return {
        "best_model": {
            "name":  meta.get("model_name"),
            "rmse":  meta.get("rmse"),
            "mae":   meta.get("mae"),
        },
        "model_comparison": meta.get("model_comparison", {}),
        "training_details": {
            "ticker":        meta.get("ticker"),
            "train_start":   meta.get("train_start"),
            "train_end":     meta.get("train_end"),
            "train_samples": meta.get("train_samples"),
            "test_samples":  meta.get("test_samples"),
            "features":      meta.get("feature_names"),
        },
    }


# ── global exception handler for anything truly unexpected ───────────

@app.exception_handler(Exception)
async def _unhandled_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
        },
    )

# Mount static files (HTML, CSS, JS) at the root. 
# FastAPI checks routes in order, so API endpoints like /predict will take priority.
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
