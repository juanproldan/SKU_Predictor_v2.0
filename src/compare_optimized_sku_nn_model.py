import os
import sqlite3
import random
import joblib
import torch
import pandas as pd

from models.sku_nn_pytorch import load_model, predict_sku

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "sku_nn")
DB_PATH = os.path.join(PROJECT_ROOT, "data", "fixacar_history.db")

# Load encoders and tokenizer
encoder_make = joblib.load(os.path.join(MODEL_DIR, 'encoder_Make.joblib'))
encoder_model_year = joblib.load(os.path.join(
    MODEL_DIR, 'encoder_Model Year.joblib'))
encoder_series = joblib.load(os.path.join(MODEL_DIR, 'encoder_Series.joblib'))
encoder_sku = joblib.load(os.path.join(MODEL_DIR, 'encoder_sku.joblib'))
tokenizer = joblib.load(os.path.join(MODEL_DIR, 'tokenizer.joblib'))

# Load optimized model
model, _ = load_model(MODEL_DIR)

# Sample 200 random rows from the DB
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql_query(
    "SELECT vin_make, vin_year, vin_series, normalized_description, sku FROM historical_parts WHERE vin_make IS NOT NULL AND vin_year IS NOT NULL AND vin_series IS NOT NULL AND normalized_description IS NOT NULL AND sku IS NOT NULL",
    conn
)
conn.close()

if len(df) > 200:
    df = df.sample(200, random_state=42)

# Run predictions and compare
correct = 0
total = 0
for idx, row in df.iterrows():
    make = str(row['vin_make'])
    year = str(row['vin_year'])
    series = str(row['vin_series'])
    desc = str(row['normalized_description'])
    true_sku = str(row['sku'])

    encoders = {
        'Make': encoder_make,
        'Model Year': encoder_model_year,
        'Series': encoder_series,
        'tokenizer': tokenizer,
        'sku': encoder_sku
    }
    pred_sku, conf = predict_sku(
        model=model,
        encoders=encoders,
        make=make,
        model_year=year,
        series=series,
        description=desc,
        device='cpu'
    )
    if pred_sku == true_sku:
        correct += 1
    total += 1

print(
    f"Optimized SKU NN Model Accuracy on {total} samples: {correct/total:.2%}")
