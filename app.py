from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
import tensorflow as tf
import requests
import joblib
import datetime
from werkzeug.utils import secure_filename
from collections import Counter

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
WEIGHTS_FOLDER = "weights"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure necessary directories exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(WEIGHTS_FOLDER):
    os.makedirs(WEIGHTS_FOLDER)

load_dotenv("sec.env")

# Meta WhatsApp API Configuration
WHATSAPP_ACCESS_TOKEN = os.getenv("WHATSAPP_ACCESS_TOKEN")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")
RECIPIENT_PHONE = os.getenv("RECIPIENT_PHONE")
WHATSAPP_API_URL = f"https://graph.facebook.com/v22.0/{PHONE_NUMBER_ID}/messages"

# Load model weights
model_paths = {
    "cnn": os.path.join(WEIGHTS_FOLDER, "cnn_model_80_20.h5"),
    "cnn_lstm": os.path.join(WEIGHTS_FOLDER, "cnn_lstm_model_80_20.h5"),
    "lstm": os.path.join(WEIGHTS_FOLDER, "lstm_model_80_20.h5"),
    "rnn": os.path.join(WEIGHTS_FOLDER, "rnn_model_80_20.h5"),
    "gradient_boosting": os.path.join(WEIGHTS_FOLDER, "gradient_boosting_80_20.pkl"),
    "random_forest": os.path.join(WEIGHTS_FOLDER, "random_forest_70_30.pkl"),
    "svm": os.path.join(WEIGHTS_FOLDER, "svm_80_20.pkl"),
    "gru": os.path.join(WEIGHTS_FOLDER, "best_gru_model_70_30.h5"),
    "xgboost": os.path.join(WEIGHTS_FOLDER, "best_xgboost_model_80_20.pkl"),
    "lightgbm": os.path.join(WEIGHTS_FOLDER, "lightgbm_model_80_20.pkl"),
}

# Load deep learning models
deep_learning_models = {}
for model_name, path in model_paths.items():
    if path.endswith(".h5") and os.path.exists(path):
        deep_learning_models[model_name] = tf.keras.models.load_model(path, compile=False)

# Load machine learning models
ml_models = {}
for model_name, path in model_paths.items():
    if path.endswith(".pkl") and os.path.exists(path):
        ml_models[model_name] = joblib.load(path)

# Class label mapping
label_mapping = {
    0: "Fall",
    1: "LFall",
    2: "Light",
    3: "RFall",
    4: "Sit",
    5: "Step",
    6: "Walk",
}

# Function to send WhatsApp alert
def send_whatsapp_message(motion_type, confidence):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    message = (
        f"⚠️ *Alert: Fall Detected!*\n"
        f"📅 *Timestamp:* {timestamp}\n"
        f"🏃 *Motion Type:* {motion_type}\n"
        f"📊 *Confidence:* {confidence:.2f}\n"
        f"\nStay safe! 🚨"
    )
    
    payload = {
        "messaging_product": "whatsapp",
        "to": RECIPIENT_PHONE,
        "type": "text",
        "text": {"body": message}
    }
    
    headers = {
        "Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }

    response = requests.post(WHATSAPP_API_URL, json=payload, headers=headers)
    
    # Debugging: Print response from WhatsApp API
    print("WhatsApp API Response:", response.status_code, response.json())
    
    if response.status_code != 200:
        print("Error sending WhatsApp message:", response.text)

# Aggregate predictions from all models
def aggregate_predictions(data):
    model_results = []

    for model_name, model in {**deep_learning_models, **ml_models}.items():
        if model_name in deep_learning_models:
            predictions = model.predict(data)
            predicted_class = np.argmax(predictions, axis=1)[0]
            confidence = np.max(predictions, axis=1)[0]
        elif model_name in ml_models:
            if hasattr(model, "predict_proba"):
                predictions = model.predict_proba(data.reshape(data.shape[0], -1))
                predicted_class = np.argmax(predictions, axis=1)[0]
                confidence = np.max(predictions, axis=1)[0]
            else:
                predictions = model.decision_function(data.reshape(data.shape[0], -1))
                predicted_class = np.argmax(predictions, axis=1)[0]
                confidence = np.abs(predictions).max()

        model_results.append({
            "model": model_name.upper(),
            "prediction": label_mapping.get(predicted_class, "Unknown"),
            "confidence": round(float(confidence), 2)
        })

    # Majority voting for final decision
    majority_label = Counter([result["prediction"] for result in model_results]).most_common(1)[0][0]
    avg_confidence = np.mean([result["confidence"] for result in model_results])

    # Send alert if fall is detected
    if majority_label in ["Fall", "LFall", "RFall"]:
        send_whatsapp_message(majority_label, avg_confidence)

    return {
        "final_prediction": majority_label,
        "confidence": round(float(avg_confidence), 2),
        "model_results": model_results
    }

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if not file.filename.endswith(".csv"):
        return jsonify({"error": "Only CSV files are allowed"}), 400

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(file.filename))
    file.save(file_path)

    df = pd.read_csv(file_path)
    data = df.to_numpy().reshape(-1, 400, 6)
    
    aggregated_result = aggregate_predictions(data)
    
    return jsonify({"result": aggregated_result})

if __name__ == "__main__":
    app.run(debug=True)
