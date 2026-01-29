import os
import hopsworks
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt 
import warnings
import json
import shutil # Added for file operations

# Machine Learning Libraries
import tensorflow as tf 
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Filter out warnings
warnings.filterwarnings('ignore')

def train_model():
    # --- 1. SECURE LOGIN ---
    api_key = os.getenv("HOPSWORKS_API_KEY")
    project = hopsworks.login(
        api_key_value=api_key,
        project="aqi_quality_fs"
    )

    # --- 2. FETCH DATA ---
    print("⏳ Fetching data from Feature Store...")
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name="karachi_aqi_features", version=7)
    
    df = fg.read()

    if 'datetime' in df.columns:
        df = df.sort_values(by='datetime').reset_index(drop=True)

    df = df.drop_duplicates().reset_index(drop=True)
    
    cols_to_drop = ["aqi_change", "aqi_pct_change", "target_aqi_24h", "datetime"]
    df_clean = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
    df_clean = df_clean.fillna(0)

    y = df_clean['aqi']
    X = df_clean.drop(columns=['aqi'])

    print(f"✅ Data Loaded. Shape: {X.shape}")

    # --- 3. SPLIT & NORMALIZE ---
    split_index = int(len(df_clean) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    scaler = MinMaxScaler(feature_range=(0,1))
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)
    
    # We will save the scaler later in the artifacts folder

    # --- 4. MODEL TRAINING LOOP ---
    models = {
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, objective="reg:squarederror"),
        "LightGBM": LGBMRegressor(n_estimators=100, verbosity=-1)
    }

    model_metrics = {}
    print("🚀 Starting Training...")
    
    for name, model in models.items():
        model.fit(X_train_norm, y_train.values.ravel())
        y_pred = model.predict(X_test_norm)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        model_metrics[name] = {"RMSE": rmse, "MAE": mae, "R2": r2}
        print(f"   👉 {name}: RMSE={rmse:.3f}, MAE={mae:.3f}, R2={r2:.3f}")

    # Train LSTM
    X_train_lstm = X_train_norm.reshape((X_train_norm.shape[0], 1, X_train_norm.shape[1]))
    X_test_lstm = X_test_norm.reshape((X_test_norm.shape[0], 1, X_test_norm.shape[1]))

    lstm_model = Sequential()
    lstm_model.add(LSTM(50, activation="relu", input_shape=(1, X_train_norm.shape[1])))
    lstm_model.add(Dense(1))
    lstm_model.compile(optimizer="adam", loss='mse')
    lstm_model.fit(X_train_lstm, y_train, epochs=10, batch_size=32, verbose=0)
    
    y_pred_lstm = lstm_model.predict(X_test_lstm).flatten()
    rmse_lstm = np.sqrt(mean_squared_error(y_test, y_pred_lstm))
    mae_lstm = mean_absolute_error(y_test, y_pred_lstm)
    r2_lstm = r2_score(y_test, y_pred_lstm)
    model_metrics["LSTM"] = {"RMSE": rmse_lstm, "MAE": mae_lstm, "R2": r2_lstm}
    print(f"   👉 LSTM: RMSE={rmse_lstm:.3f} MAE={mae:.3f}, R2={r2:.3f}")

    # --- 5. SELECT BEST MODEL ---
    best_model_name = min(model_metrics, key=lambda x: model_metrics[x]["RMSE"])
    print(f"\n🏆 Best Model: {best_model_name}")

    # --- 6. PREPARE ARTIFACTS FOLDER ---
    # Create a temporary directory to hold everything we want to upload
    artifact_dir = "model_artifacts"
    if os.path.exists(artifact_dir):
        shutil.rmtree(artifact_dir) # Clean up old runs
    os.makedirs(artifact_dir)

    # Save Best Model into that folder
    if best_model_name == "LSTM":
        lstm_model.save(os.path.join(artifact_dir, "best_model.keras"))
    else:
        best_model = models[best_model_name]
        joblib.dump(best_model, os.path.join(artifact_dir, "best_model.pkl"))

    # Save Scaler into that folder
    joblib.dump(scaler, os.path.join(artifact_dir, "scaler.pkl"))
    print("✅ Model and Scaler saved locally to 'model_artifacts/'")

    # --- 7. UPLOAD TO REGISTRY ---
    mr = project.get_model_registry()
    
    input_example = X_train_norm[:1]

    # Create the Model Object (Metadata)
    if best_model_name == "LSTM":
        hw_model = mr.tensorflow.create_model(
            name="karachi_aqi_best_model",
            metrics=model_metrics[best_model_name],
            input_example=input_example,
            description=f"Best AQI Model: {best_model_name}"
        )
    else:
        hw_model = mr.sklearn.create_model(
            name="karachi_aqi_best_model",
            metrics=model_metrics[best_model_name],
            input_example=input_example,
            description=f"Best AQI Model: {best_model_name}"
        )

    # UPLOAD THE ENTIRE DIRECTORY (This fixes your error!)
    print("Uploading Model artifacts to Hopsworks...")
    hw_model.save(artifact_dir)
    
    print("✅ Training Pipeline Complete! Model Version created successfully.")

if __name__ == "__main__":
    train_model()



