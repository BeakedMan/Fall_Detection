from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv
import tensorflow as tf
import requests
import datetime
import joblib
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
WEIGHTS_FOLDER = "weights"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(WEIGHTS_FOLDER, exist_ok=True)

# Load StandardScaler used during training
scaler = joblib.load(os.path.join(WEIGHTS_FOLDER, "scaler.pkl"))

# Location storage
user_location = {"latitude": None, "longitude": None}

# Load environment variables
load_dotenv("sec.env")
WHATSAPP_ACCESS_TOKEN = os.getenv("WHATSAPP_ACCESS_TOKEN")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")
RECIPIENT_PHONE = os.getenv("RECIPIENT_PHONE")
WHATSAPP_API_URL = f"https://graph.facebook.com/v17.0/{PHONE_NUMBER_ID}/messages"

# Load expert + gating models
cnn = tf.keras.models.load_model(os.path.join(WEIGHTS_FOLDER, "cnn_model.h5"))
lstm = tf.keras.models.load_model(os.path.join(WEIGHTS_FOLDER, "lstm_model.h5"))
gru = tf.keras.models.load_model(os.path.join(WEIGHTS_FOLDER, "gru_model.h5"))
cnn_lstm = tf.keras.models.load_model(os.path.join(WEIGHTS_FOLDER, "cnn_lstm_model.h5"))
resnet = tf.keras.models.load_model(os.path.join(WEIGHTS_FOLDER, "resnet_model.h5"))
bilstm = tf.keras.models.load_model(os.path.join(WEIGHTS_FOLDER, "bilstm_model.h5"))
gating_model = tf.keras.models.load_model(os.path.join(WEIGHTS_FOLDER, "gating_model.h5"))

# Label Mapping
label_mapping = {
    0: "Fall", 1: "LFall", 2: "Light", 3: "RFall",
    4: "Sit", 5: "Step", 6: "Walk",
}

# Send WhatsApp Alert
def send_whatsapp_message(motion_type, confidence):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    location_message = ""
    if user_location["latitude"] and user_location["longitude"]:
        location_message = f"\n📍 Location: https://www.google.com/maps?q={user_location['latitude']},{user_location['longitude']}"
    
    message = (
        f"⚠️ Alert: Fall Detected!\n"
        f"📅 Timestamp: {timestamp}\n"
        f"🏃 Motion Type: {motion_type}\n"
        f"📊 Confidence: {confidence:.2f}\n"
        f"{location_message}\n"
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
    
    print(f"WhatsApp API Response ({response.status_code}):", response.json())

    if response.status_code != 200:
        print("❌ Error sending WhatsApp message:", response.text)

# Predict using full MoE setup
def predict_with_moe(data):
    cnn_pred = cnn.predict(data)
    lstm_pred = lstm.predict(data)
    gru_pred = gru.predict(data)
    cnn_lstm_pred = cnn_lstm.predict(data)
    resnet_pred = resnet.predict(data)
    bilstm_pred = bilstm.predict(data)

    gate_input = np.concatenate([cnn_pred, lstm_pred, gru_pred, cnn_lstm_pred, resnet_pred, bilstm_pred], axis=1)
    gating_pred = gating_model.predict(gate_input)

    final_class = np.argmax(gating_pred, axis=1)[0]
    confidence = np.max(gating_pred)
    label = label_mapping[final_class]

    if label in ["Fall", "LFall", "RFall", "Light"]:
        send_whatsapp_message(label, confidence)

    return label, round(float(confidence), 2), [cnn_pred, lstm_pred, gru_pred, cnn_lstm_pred, resnet_pred, bilstm_pred]

@app.route("/",  methods=["GET", "POST"])
def index():
    return render_template("index.html")

@app.route("/latest_prediction", methods=["GET"])
def latest_prediction():
    file_path = "sensor_data.csv"
    if not os.path.exists(file_path):
        return jsonify({"error": "No sensor data found"}), 404

    df = pd.read_csv(file_path)
    if "Processed" not in df.columns:
        df["Processed"] = "NO"

    if all(df["Processed"] == "YES"):
        return jsonify({"error": "Waiting for new data..."})

    if df.shape[0] != 400 or df.shape[1] < 6:
        return jsonify({"error": "Invalid CSV format. Ensure 400 rows and 6 columns."}), 400

    data = df.iloc[:, :6].to_numpy().reshape(-1, 400, 6)
    
    # Standardize the data
    data_flat = data.reshape(1, -1)  # shape (1, 2400)
    data_scaled = scaler.transform(data_flat)
    data = data_scaled.reshape(-1, 400, 6)  # shape (1, 400, 6)

    final_prediction, confidence, expert_preds = predict_with_moe(data)

    df["Processed"] = "YES"
    df.to_csv(file_path, index=False)

    return jsonify({
        "result": {
            "final_prediction": final_prediction,
            "confidence": confidence,
            "raw_data": df.iloc[:, :6].values.tolist()  # Send 400x6 readings
        }
    })

@app.route("/update_location", methods=["POST"])
def update_location():
    """Updates user location when received from frontend"""
    global user_location
    try:
        data = request.get_json()
        user_location["latitude"] = data.get("latitude")
        user_location["longitude"] = data.get("longitude")
        return jsonify({"message": "Location updated successfully!"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

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
    if df.shape[0] != 400 or df.shape[1] < 6:
        return jsonify({"error": "Invalid CSV format. Ensure 400 rows and 6 columns."}), 400

    df["Processed"] = "NO"
    df.to_csv(file_path, index=False)

    return jsonify({"message": "New data uploaded successfully!"})

if __name__ == "__main__":
    app.run(debug=True)
