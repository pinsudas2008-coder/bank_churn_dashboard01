# backend/api.py
from flask import Flask, request, jsonify
import joblib, os, numpy as np

app = Flask(__name__)

# --- Paths ---
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(BASE, "models")
RF_PATH = os.path.join(MODELS_DIR, "rf_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")

# --- Load models ---
rf = joblib.load(RF_PATH) if os.path.exists(RF_PATH) else None
scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None

# --- Home route ---
@app.route("/")
def home():
    return "Bank Churn Dashboard API is running!"

# --- Health check ---
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": rf is not None})

# --- Predict route ---
@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True)
    features = payload.get("features", None)
    if features is None:
        return jsonify({"error": "send JSON { 'features': [..] }"}), 400

    try:
        X = np.array(features).reshape(1, -1)
        if scaler is not None:
            X = scaler.transform(X)
        if rf is None:
            return jsonify({"error": "no model found. Run model_train.py"}), 500

        proba = float(rf.predict_proba(X)[0, 1])
        pred = int(proba >= 0.5)
        return jsonify({"prediction": pred, "probability": proba})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Main ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
