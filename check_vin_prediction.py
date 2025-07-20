#!/usr/bin/env python3
"""
Check what the VIN prediction system returns for our test VIN.
"""

import sys
import os
import joblib
import numpy as np
import pandas as pd

sys.path.append('src')

try:
    from train_vin_predictor import extract_vin_features, decode_year
except ImportError:
    print("❌ Could not import VIN prediction functions")
    exit(1)

def check_vin_prediction():
    """Check VIN prediction for our test case."""

    test_vin = "3MDDJ25AALM219739"

    print(f"=== VIN PREDICTION ANALYSIS ===")
    print(f"Testing VIN: {test_vin}")
    print()

    try:
        # Load models (same as main_app.py)
        MODEL_DIR = 'models'

        print("Loading VIN prediction models...")
        model_maker = joblib.load(os.path.join(MODEL_DIR, 'vin_maker_model.joblib'))
        encoder_x_maker = joblib.load(os.path.join(MODEL_DIR, 'vin_maker_encoder_x.joblib'))

        model_year = joblib.load(os.path.join(MODEL_DIR, 'vin_year_model.joblib'))
        encoder_x_year = joblib.load(os.path.join(MODEL_DIR, 'vin_year_encoder_x.joblib'))

        model_series = joblib.load(os.path.join(MODEL_DIR, 'vin_series_model.joblib'))
        encoder_x_series = joblib.load(os.path.join(MODEL_DIR, 'vin_series_encoder_x.joblib'))

        print("Models loaded successfully.")
        print()

        # Extract features and predict (same logic as main_app.py)
        features = extract_vin_features(test_vin)
        if not features:
            print("❌ Could not extract features from VIN")
            return

        print(f"Extracted features: {features}")

        # Predict Make
        X_maker = encoder_x_maker.transform([features])
        predicted_make = model_maker.predict(X_maker)[0]

        # Predict Year
        X_year = encoder_x_year.transform([features])
        predicted_year_encoded = model_year.predict(X_year)[0]
        predicted_year = decode_year(predicted_year_encoded)

        # Predict Series
        X_series = encoder_x_series.transform([features])
        predicted_series = model_series.predict(X_series)[0]

        print("🔍 VIN Prediction Results:")
        print(f"   Make: {predicted_make}")
        print(f"   Year: {predicted_year}")
        print(f"   Series: {predicted_series}")
        print()

        # Check if this matches our database entries
        if predicted_make.upper() == 'MAZDA':
            print("✅ Make prediction matches database entries (MAZDA)")
        else:
            print(f"❌ Make prediction mismatch: predicted '{predicted_make}' but database has 'MAZDA'")
            print("   This explains why no matches are found!")

        print()
        print("=== RECOMMENDATIONS ===")
        if predicted_make.upper() != 'MAZDA':
            print("1. 🔧 VIN prediction is incorrect - this VIN should predict MAZDA")
            print("2. 🔧 Check VIN prediction training data")
            print("3. 🔧 Consider manual VIN correction for testing")
        else:
            print("1. ✅ VIN prediction is correct")
            print("2. 🔍 Issue might be in Year/Series matching")
            print("3. 🔍 Check confidence scoring algorithm")

    except Exception as e:
        print(f"❌ Error in VIN prediction: {e}")
        print("   VIN predictor might not be properly initialized")

if __name__ == "__main__":
    check_vin_prediction()
