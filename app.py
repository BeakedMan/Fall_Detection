from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow warnings


import tensorflow as tf
from werkzeug.utils import secure_filename
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
WEIGHTS_FOLDER = 'weights'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure necessary directories exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(WEIGHTS_FOLDER):
    os.makedirs(WEIGHTS_FOLDER)

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
    "lightgbm": os.path.join(WEIGHTS_FOLDER, "lightgbm_model_80_20.pkl")
}

custom_objects = {
    "accuracy": tf.keras.metrics.Accuracy(),
    "categorical_crossentropy": tf.keras.losses.CategoricalCrossentropy()
}

# Load deep learning models
deep_learning_models = {}
for model_name, path in model_paths.items():
    if path.endswith(".h5") and os.path.exists(path):
        deep_learning_models[model_name] = tf.keras.models.load_model(path, compile=False)

# Load machine learning models
import joblib
ml_models = {}
for model_name, path in model_paths.items():
    if path.endswith(".pkl") and os.path.exists(path):
        ml_models[model_name] = joblib.load(path)

# Data preprocessing function
def preprocess_csv(file_path):
    df = pd.read_csv(file_path)  # Skip the first row (column names)

    print(df)
    
    # Ensure the CSV contains exactly 400 rows per instance
    if df.shape[0] % 400 != 0 or df.shape[1] < 6:
        return None

    # Reshape the data to (-1, 400, 6) since each instance has 400 entries
    data = df.iloc[:, :6].to_numpy().reshape(-1, 400, 6)

    # Apply StandardScaler normalization
    #scaler = StandardScaler()
    #data = scaler.fit_transform(data.reshape(-1, 400 * 6)).reshape(-1, 400, 6)
    print("Processed Data for Model:\n", data)
    return data

# Prediction function
def predict_from_model(data, model_name):
    if model_name in deep_learning_models:
        model = deep_learning_models[model_name]
        predictions = model.predict(data)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions, axis=1)[0]
        print(predictions)
    elif model_name in ml_models:
        model = ml_models[model_name]
        if hasattr(model, "predict_proba"):
            predictions = model.predict_proba(data.reshape(data.shape[0], -1))
            predicted_class = np.argmax(predictions, axis=1)[0]
            confidence = np.max(predictions, axis=1)[0]
            print(predictions)
        else:
            predictions = model.decision_function(data.reshape(data.shape[0], -1))
            predicted_class = np.argmax(predictions, axis=1)[0]
            confidence = np.abs(predictions).max()  # Approximate confidence
            print(predictions)
    else:
        return None

    # Mapping class labels to names
    label_mapping = {
        0: "Fall",
        1: "LFall",
        2: "Light",
        3: "RFall",
        4: "Sit",
        5: "Step",
        6: "Walk"
    }
    return {"class": label_mapping.get(predicted_class, "Unknown"), "confidence": round(float(confidence), 2)}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    model = request.form.get("model")

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if not file.filename.endswith('.csv'):
        return jsonify({"error": "Only CSV files are allowed"}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
    file.save(file_path)

    # Process CSV
    data = preprocess_csv(file_path)
    
    if data is None:
        return jsonify({"error": "Invalid CSV format. Ensure 400 rows per sample and 6 columns."}), 400

    prediction = predict_from_model(data, model)
    if prediction is None:
        return jsonify({"error": "Model not found"}), 400

    return jsonify({"result": prediction})

if __name__ == '__main__':
    app.run(debug=True)
