import sys
sys.path.append('src')

from train_vin_predictor import extract_vin_features, decode_year
import joblib
import os

# Load the models
MODEL_DIR = 'models'
model_maker = joblib.load(os.path.join(MODEL_DIR, 'vin_maker_model.joblib'))
model_year = joblib.load(os.path.join(MODEL_DIR, 'vin_year_model.joblib'))
model_series = joblib.load(os.path.join(MODEL_DIR, 'vin_series_model.joblib'))

# Test the VIN that was used
test_vin = '9FCDF5553F0112445'
print(f"Testing VIN: {test_vin}")

# Extract features
features = extract_vin_features(test_vin)
print(f"VIN Features: {features}")

# Predict
make_pred = model_maker.predict([features])[0]
year_pred = model_year.predict([features])[0]
series_pred = model_series.predict([features])[0]

print(f"Predicted Make: {make_pred}")
print(f"Predicted Year: {decode_year(year_pred)}")
print(f"Predicted Series: {series_pred}")

# Check if we can find similar series in database
import sqlite3
conn = sqlite3.connect('data/fixacar_history.db')
cursor = conn.cursor()

# Look for series that contain the predicted series
cursor.execute("SELECT DISTINCT vin_series FROM historical_parts WHERE vin_make = ? AND vin_year = ? AND vin_series LIKE ?", 
               (make_pred, decode_year(year_pred), f'%{series_pred}%'))
similar_series = cursor.fetchall()
print(f"Database series containing '{series_pred}': {similar_series}")

conn.close()
