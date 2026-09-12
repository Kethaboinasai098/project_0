import os
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.linear_model import TheilSenRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor

app = Flask(__name__)

MODELS = {}
RESULTS_DF = None
SCALER = None
X_COLUMNS = None
BEST_MODEL_NAME = "CatBoost Regressor"
BEST_MODEL = None

def initialize_ml_pipeline():
    global MODELS, RESULTS_DF, SCALER, X_COLUMNS, BEST_MODEL, BEST_MODEL_NAME
    
    dataset_path = "dataset/Student_Performance999.csv"
    if not os.path.exists(dataset_path):
        print(f"[Warning] Dataset not found at {dataset_path}. Please place your CSV there.")
        return

    print("[INFO] Loading Student Performance Dataset...")
    df = pd.read_csv(dataset_path)
    
    # Preprocessing
    df = df.dropna()
    df = pd.get_dummies(df, drop_first=True)
    target_column = "overall_score"
    
    if target_column not in df.columns:
        target_column = df.columns[-1]
        
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    X_COLUMNS = X.columns.tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    global SCALER
    SCALER = scaler
    
    models_dict = {
        "CatBoost Regressor": CatBoostRegressor(verbose=0, random_state=42),
        "LightGBM Regressor": LGBMRegressor(random_state=42),
        "MLP Regressor": MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
        "Theil-Sen Regressor": TheilSenRegressor(random_state=42),
        "LinearSVR": LinearSVR(random_state=42, max_iter=5000)
    }
    
    evaluation_results = []
    print("[INFO] Training regression algorithms...")
    for name, model in models_dict.items():
        if name in ["MLP Regressor", "LinearSVR"]:
            model.fit(X_train_scaled, y_train)
            test_preds = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            test_preds = model.predict(X_test)
            
        r2 = r2_score(y_test, test_preds)
        rmse = np.sqrt(mean_squared_error(y_test, test_preds))
        mae = mean_absolute_error(y_test, test_preds)
        
        evaluation_results.append({
            "Model": name,
            "R2 Score": round(r2, 6),
            "RMSE": round(rmse, 6),
            "MAE": round(mae, 6)
        })
        MODELS[name] = model
        
    global RESULTS_DF
    RESULTS_DF = pd.DataFrame(evaluation_results).sort_values(by="R2 Score", ascending=False)
    BEST_MODEL_NAME = RESULTS_DF.iloc[0]["Model"]
    BEST_MODEL = MODELS[BEST_MODEL_NAME]
    print(f"[INFO] Pipeline Ready! Best Model: {BEST_MODEL_NAME}")

initialize_ml_pipeline()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    if RESULTS_DF is None:
        return jsonify({"error": "Models not initialized."}), 500
    return jsonify(RESULTS_DF.to_dict(orient="records"))

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        model_name = data.get("model", BEST_MODEL_NAME)
        model = MODELS.get(model_name, BEST_MODEL)
        
        input_data = {col: 0.0 for col in X_COLUMNS}
        
        if "study_hours" in input_data: input_data["study_hours"] = float(data.get("study_hours", 15))
        if "attendance_percentage" in input_data: input_data["attendance_percentage"] = float(data.get("attendance_percentage", 85))
        if "math_score" in input_data: input_data["math_score"] = float(data.get("math_score", 75))
        if "science_score" in input_data: input_data["science_score"] = float(data.get("science_score", 75))
        if "english_score" in input_data: input_data["english_score"] = float(data.get("english_score", 75))
        if "age" in input_data: input_data["age"] = float(data.get("age", 16))
        
        parent_edu = data.get("parent_education", "graduate")
        edu_key = f"parent_education_{parent_edu}"
        if edu_key in input_data: input_data[edu_key] = 1
        
        study_meth = data.get("study_method", "notes")
        meth_key = f"study_method_{study_meth}"
        if meth_key in input_data: input_data[meth_key] = 1
        
        df_pred = pd.DataFrame([input_data])
        
        if model_name in ["MLP Regressor", "LinearSVR"]:
            df_scaled = SCALER.transform(df_pred)
            pred_score = float(model.predict(df_scaled)[0])
        else:
            pred_score = float(model.predict(df_pred)[0])
            
        pred_score = max(0.0, min(100.0, pred_score))
        
        if pred_score >= 75:
            risk = "Low Academic Risk (On Track)"
            risk_class = "success"
        elif pred_score >= 55:
            risk = "Moderate Risk (Needs Support)"
            risk_class = "warning"
        else:
            risk = "High Risk (Immediate Intervention Required)"
            risk_class = "danger"
            
        return jsonify({
            "predicted_score": round(pred_score, 2),
            "risk_assessment": risk,
            "risk_class": risk_class,
            "model_used": model_name
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)