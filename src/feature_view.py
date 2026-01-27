import hopsworks

# 1. Login
project = hopsworks.login()
fs = project.get_feature_store()

# 2. Get the Feature Group
fg = fs.get_feature_group(name="karachi_aqi_features", version=7)

# 3. Select all features and create the Feature View
# We tell Hopsworks that 'aqi' is the label we want to predict
query = fg.select_all()

try:
    feature_view = fs.create_feature_view(
        name="aqi_prediction_view",
        description="View for Karachi AQI prediction model",
        labels=["aqi"],
        query=query
    )
    print("✅ Feature View 'aqi_prediction_view' created successfully!")
except:
    print("ℹ️ Feature View already exists or an error occurred.")