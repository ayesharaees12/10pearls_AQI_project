# --- IMPORTS ---
import os
import hopsworks
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt 
import warnings

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

# Filter out annoying warnings
warnings.filterwarnings('ignore')

# --- 1. SECURE LOGIN ---
# This looks for the secret key (GitHub Actions) or uses your local key (Laptop)
api_key = os.getenv("HOPSWORKS_API_KEY")

if not api_key:
    # TODO: Replace this string with your actual API key if running locally!
    api_key = "DGdCrO6zZkBc42ad.L1OpxZ0oXVXn9Md1YPllyCIu8XOj0Q3epPmrVZeoJikQ9mNdJTe9uqtsZAQQ93FJ" 

project = hopsworks.login(
    api_key_value=api_key,
    project="aqi_quality_fs"
)

# --- 2. FETCH DATA FROM FEATURE STORE ---
print("⏳ Fetching data from Feature Store...")
fs = project.get_feature_store()

fg = fs.get_feature_group(
    name="karachi_aqi_features",
    version=7
)

# Read the data
df = fg.read(read_options={"use_hive":True,"arrow_flight":False})

# CRITICAL FIX: Ensure data is sorted by time for Time Series training
# Replace 'datetime' with your actual date column name if it's different
if 'datetime' in df.columns:
    df = df.sort_values(by='datetime').reset_index(drop=True)

print("✅ Data successfully loaded!")
print(df.head(5)) # Use print() so you can see it in the script output

#Drop duplicates 
df=df.drop_duplicates().reset_index(drop=True)
#Remove leakage columns
df.drop(columns=["aqi_change","aqi_pct_change","date_time","target_aqi_24h"],inplace=True)

df.columns

#Define Feature and Target 
X = df.drop(columns=['aqi', 'datetime'] , errors='ignore')
y = df['aqi']
df_model = pd.concat([X, y], axis=1).dropna().reset_index(drop=True)

#Time BASED TRAIN TEST SPLIT 
#Define Split_Index
split_index=int(len(df)*0.8)


# Time-based Split (80% train, 20% test) - Important for time-series
X_train=X.iloc[:split_index]
X_test=X.iloc[split_index:]

y_train=y.iloc[:split_index]
y_test=y.iloc[split_index:]
print(f"Train Shape:{X_train.shape},Test Shape:{X_test.shape}")
#Fill NaN values before scaling
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

#Normalize Data Using MIN-MAX scaler 
scaler=MinMaxScaler(feature_range=(0,1))
# Fit on training data only (never fit on test data!)
X_train_norm=scaler.fit_transform(X_train)
# Transform test data using the same scaler
X_test_norm=scaler.transform(X_test)
# Save the fitted scaler for later use (inference/prediction)
joblib.dump(scaler,"scaler.pkl")
print(f"MinMaxScaler Applied Successfully:")
print(f"X_train_norm min_max:",X_train_norm.min(), X_train_norm.max())

# MODEL TRAINING AND THEIR METRICS
models={
    "Ridge":Ridge(alpha=1.0),
    "RandomForest":RandomForestRegressor(n_estimators=200,random_state=42,max_depth=5),
    "XGBoost":XGBRegressor(n_estimators=200,random_state=42,objective="reg:squarederror",
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        reg_lambda=1
),
    "LightGBM":LGBMRegressor(n_estimators=200,random_state=42,verbosity=-1,force_row_wise=True)
}
model_metrics={}
for name,model in models.items():
    model.fit(X_train_norm,y_train.values.ravel())
    y_pred=model.predict(X_test_norm)
    rmse=np.sqrt(mean_squared_error(y_test,y_pred))
    mae=mean_absolute_error(y_test,y_pred)
    r2=r2_score(y_test,y_pred)
    model_metrics[name]={"RMSE":rmse, "MAE":mae, "R2":r2}
    print(f"{name}  RMSE:{rmse:.3f}, MAE: {mae:.3f}, R2:{r2:.3f}")
#DEEP LEARNING MODEL:LSTM
# Reshape for LSTM: [samples, timesteps=1, features]
X_train_lstm=X_train_norm.reshape((X_train_norm.shape[0],1,X_train_norm.shape[1]))
X_test_lstm=X_test_norm.reshape((X_test_norm.shape[0],1,X_test_norm.shape[1]))

lstm_model=Sequential()
lstm_model.add(LSTM(50,activation="relu",input_shape=(1,X_train_norm.shape[1])))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer="adam",loss='mse')

lstm_model.fit(X_train_lstm,y_train,epochs=20,batch_size=32,validation_split=0.1,verbose=1)
y_pred_lstm=lstm_model.predict(X_test_lstm).flatten()

rmse_lstm=np.sqrt(mean_squared_error(y_test,y_pred_lstm))
mae_lstm=mean_absolute_error(y_test,y_pred_lstm)
r2_lstm=r2_score(y_test,y_pred_lstm)

model_metrics["lstm_model"]={"RMSE":rmse_lstm ,"MAE":mae_lstm, "R2":r2_lstm}
print(f"LSTM Metrics: RMSE={rmse_lstm:.2f}, MAE={mae_lstm:.2f}, R2={r2_lstm:.2f}")

#MODEL SAVE INTO REGISTRY 
import hopsworks
import joblib

# Login
project = hopsworks.login(
    api_key_value="DGdCrO6zZkBc42ad.L1OpxZ0oXVXn9Md1YPllyCIu8XOj0Q3epPmrVZeoJikQ9mNdJTe9uqtsZAQQ93FJ",
    project="aqi_quality_fs"
)


# 1. Identify the best model from your metrics
best_model_name = min(model_metrics, key=lambda x: model_metrics[x]["RMSE"])
# KEY FIX: Assign the actual model object to 'best_model'
best_model = models[best_model_name] 

print(f"🏆 Best Model Identified: {best_model_name}")
print(f"📊 Metrics: {model_metrics[best_model_name]}")

MODEL_NAME = "karachi_aqi_best_model"

# 2. Ensure your 'model_registry' folder exists
registry_folder = "model_registry"
if not os.path.exists(registry_folder):
    os.makedirs(registry_folder)

# 3. Save the Best Model LOCALLY
if best_model_name == "LSTM":
    local_model_path = os.path.join(registry_folder, "best_model_lstm.keras")
    best_model.save(local_model_path)
    best_model.save("model.keras")
    upload_file = "model.keras"
else:
    local_model_path = os.path.join(registry_folder, f"best_model_{best_model_name.lower()}.pkl")
    joblib.dump(best_model, local_model_path)
    joblib.dump(best_model, "model.pkl")
    upload_file = "model.pkl"



# 5. REGISTER MODEL IN HOPSWORKS
mr = project.get_model_registry()

if best_model_name == "LSTM":
    hw_model = mr.tensorflow.create_model(
        name=MODEL_NAME,
        metrics=model_metrics[best_model_name],
        description=f"Best AQI model ({best_model_name})"
    )
else:
    hw_model = mr.sklearn.create_model(
        name=MODEL_NAME,
        metrics=model_metrics[best_model_name],
        description=f"Best AQI model ({best_model_name})"
    )

hw_model.save(upload_file)

# --- SHAP SECTION ---

# 1. Convert normalized arrays back to DataFrames
X_test_norm_df = pd.DataFrame(X_test_norm, columns=X_test.columns)

# 2. SHAP Logic (Using the best_model object we defined at the top)
if best_model_name in ["RandomForest", "XGBoost", "LightGBM"]:
    print(f"Generating SHAP values for {best_model_name}...")
    
    # TreeExplainer needs the model object (best_model)
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_test_norm_df)

    # List handling for specific model versions
    if isinstance(shap_values, list) and len(shap_values) > 0:
        shap_values = shap_values[0]

    # 3. Plotting
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values, 
        X_test_norm_df, 
        plot_type="bar" 
    )
    plt.show()
else:
    print(f"SHAP summary not generated for {best_model_name}.")

# Cleanup temporary files
if os.path.exists(upload_file):
    os.remove(upload_file)

print(f"✅ Local model saved to: {local_model_path}")



