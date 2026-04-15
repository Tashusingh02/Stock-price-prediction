import requests
import json
from datetime import datetime, timedelta

BASE_URL = "http://127.0.0.1:8000"

def test_health():
    print("Testing / ...")
    r = requests.get(f"{BASE_URL}/")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["model_ready"] == True
    print("PASS: Health check")

def test_model_info():
    print("Testing /model-info ...")
    r = requests.get(f"{BASE_URL}/model-info")
    assert r.status_code == 200
    data = r.json()
    assert "best_model" in data
    assert "model_comparison" in data
    assert "training_details" in data
    assert "Trend" in data["training_details"]["features"]
    print(f"PASS: Model info (Best: {data['best_model']['name']})")

def test_predict_aapl():
    print("Testing /predict?ticker=AAPL ...")
    r = requests.get(f"{BASE_URL}/predict?ticker=AAPL")
    assert r.status_code == 200
    data = r.json()
    
    # 7 predictions exist
    assert len(data["predictions"]) == 7
    
    # Values not identical
    prices = [p["predicted_price"] for p in data["predictions"]]
    assert len(set(prices)) > 1, f"Prices are repeating: {prices}"
    
    # Dates valid and skip weekends
    for i in range(len(data["predictions"])):
        dt = datetime.strptime(data["predictions"][i]["date"], "%Y-%m-%d")
        assert dt.weekday() < 5, f"Prediction date is a weekend: {data['predictions'][i]['date']}"
        if i > 0:
            prev_dt = datetime.strptime(data["predictions"][i-1]["date"], "%Y-%m-%d")
            assert dt > prev_dt
            
    # Logic consistency
    avg = data["average_predicted_price"]
    curr = data["current_price"]
    if avg > curr:
        assert data["signal"] == "BUY"
        assert "Upward" in data["reason"]
    else:
        assert data["signal"] == "SELL"
        assert "Downward" in data["reason"]
        
    # Confidence and Risk
    assert 0 <= data["confidence"] <= 1
    assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
    
    print(f"PASS: AAPL prediction (Signal: {data['signal']}, Risk: {data['risk_level']}, Confidence: {data['confidence']})")

def test_case_normalization():
    print("Testing normalization (aapl) ...")
    r = requests.get(f"{BASE_URL}/predict?ticker=aapl")
    assert r.status_code == 200
    assert r.json()["ticker"] == "AAPL"
    print("PASS: Ticker normalization")

def test_invalid_ticker():
    print("Testing invalid ticker (!!!) ...")
    r = requests.get(f"{BASE_URL}/predict?ticker=!!!")
    assert r.status_code == 400
    assert "error" in r.json()["detail"]
    print("PASS: Invalid ticker error handling")

if __name__ == "__main__":
    try:
        test_health()
        test_model_info()
        test_predict_aapl()
        test_case_normalization()
        test_invalid_ticker()
        print("\nALL TESTS PASSED SUCCESSFULLY!")
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
