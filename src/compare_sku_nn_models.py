import os
import sqlite3
import random
import numpy as np
import torch
import joblib
from models.sku_nn_pytorch import OptimizedSKUNNModel, predict_sku

# --- Configuration ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, 'data', 'fixacar_history.db')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models', 'sku_nn')
SAMPLE_SIZE = 200  # Number of samples to test

# --- Load sample data from DB ---


def load_sample_data(db_path, sample_size):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT vin_number, normalized_description, sku FROM historical_parts WHERE sku IS NOT NULL")
    rows = cursor.fetchall()
    conn.close()
    if len(rows) > sample_size:
        rows = random.sample(rows, sample_size)
    return rows

# --- Load encoders and tokenizer ---


def load_encoders_and_tokenizer(model_dir):
    """Load encoders and tokenizer only (no model)."""
    encoders = {}
    encoders['Make'] = joblib.load(
        os.path.join(model_dir, 'encoder_Make.joblib'))
    encoders['Model Year'] = joblib.load(
        os.path.join(model_dir, 'encoder_Model Year.joblib'))
    encoders['Series'] = joblib.load(
        os.path.join(model_dir, 'encoder_Series.joblib'))
    encoders['sku'] = joblib.load(
        os.path.join(model_dir, 'encoder_sku.joblib'))
    try:
        encoders['tokenizer'] = joblib.load(
            os.path.join(model_dir, 'tokenizer.joblib'))
    except Exception:
        from utils.pytorch_tokenizer import PyTorchTokenizer
        encoders['tokenizer'] = PyTorchTokenizer(
            num_words=10000, oov_token="<OOV>")
    return encoders

# --- Load optimized model ---


def load_optimized_model(model_dir, encoders):
    cat_input_size = 3
    vocab_size = len(encoders['tokenizer'].word_index) + 1
    num_classes = len(encoders['sku'].classes_)
    model = OptimizedSKUNNModel(
        cat_input_size=cat_input_size,
        vocab_size=vocab_size,
        embedding_dim=256,
        hidden_size=256,
        num_classes=num_classes
    )
    model_path = os.path.join(model_dir, 'sku_nn_model_pytorch_optimized.pth')
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

# --- Predict VIN details using encoders (from training scripts) ---


def predict_vin_details(vin, encoders):
    # Use the same logic as in training scripts for fair comparison
    try:
        from train_vin_predictor import extract_vin_features
        VIN_MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
        model_vin_maker = joblib.load(os.path.join(
            VIN_MODEL_DIR, 'vin_maker_model.joblib'))
        encoder_x_vin_maker = joblib.load(os.path.join(
            VIN_MODEL_DIR, 'vin_maker_encoder_x.joblib'))
        encoder_y_vin_maker = joblib.load(os.path.join(
            VIN_MODEL_DIR, 'vin_maker_encoder_y.joblib'))
        model_vin_year = joblib.load(os.path.join(
            VIN_MODEL_DIR, 'vin_year_model.joblib'))
        encoder_x_vin_year = joblib.load(os.path.join(
            VIN_MODEL_DIR, 'vin_year_encoder_x.joblib'))
        encoder_y_vin_year = joblib.load(os.path.join(
            VIN_MODEL_DIR, 'vin_year_encoder_y.joblib'))
        model_vin_series = joblib.load(os.path.join(
            VIN_MODEL_DIR, 'vin_series_model.joblib'))
        encoder_x_vin_series = joblib.load(os.path.join(
            VIN_MODEL_DIR, 'vin_series_encoder_x.joblib'))
        encoder_y_vin_series = joblib.load(os.path.join(
            VIN_MODEL_DIR, 'vin_series_encoder_y.joblib'))
        features = extract_vin_features(vin)
        details = {'Make': 'N/A', 'Model Year': 'N/A', 'Series': 'N/A'}
        if features:
            try:
                wmi_encoded = encoder_x_vin_maker.transform(
                    [[features['wmi']]])
                if -1 not in wmi_encoded:
                    pred_encoded = model_vin_maker.predict(wmi_encoded)
                    if pred_encoded[0] != -1:
                        details['Make'] = encoder_y_vin_maker.inverse_transform(
                            pred_encoded.reshape(-1, 1))[0]
                year_code_encoded = encoder_x_vin_year.transform(
                    [[features['year_code']]])
                if -1 not in year_code_encoded:
                    pred_encoded = model_vin_year.predict(year_code_encoded)
                    if pred_encoded[0] != -1:
                        details['Model Year'] = encoder_y_vin_year.inverse_transform(
                            pred_encoded.reshape(-1, 1))[0]
                series_features_encoded = encoder_x_vin_series.transform(
                    [[features['wmi'], features['vds_full']]])
                if -1 not in series_features_encoded[0]:
                    pred_encoded = model_vin_series.predict(
                        series_features_encoded)
                    if pred_encoded[0] != -1:
                        details['Series'] = encoder_y_vin_series.inverse_transform(
                            pred_encoded.reshape(-1, 1))[0]
            except Exception:
                pass
        return details
    except Exception:
        return {'Make': 'N/A', 'Model Year': 'N/A', 'Series': 'N/A'}

# --- Main evaluation logic ---


def main():
    print("Loading sample data...")
    samples = load_sample_data(DB_PATH, SAMPLE_SIZE)
    print(f"Loaded {len(samples)} samples.")
    print("Loading encoders and tokenizer...")
    encoders = load_encoders_and_tokenizer(MODEL_DIR)
    print("Loading optimized model...")
    opt_model = load_optimized_model(MODEL_DIR, encoders)
    if not opt_model:
        print("Error: Could not load optimized model for evaluation.")
        return
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = []
    for vin, desc, true_sku in samples:
        vin_details = predict_vin_details(vin, encoders)
        make, year, series = vin_details['Make'], vin_details['Model Year'], vin_details['Series']
        opt_pred, opt_conf = predict_sku(
            opt_model, encoders, make, year, series, desc, device)
        results.append({
            'vin': vin,
            'desc': desc,
            'true_sku': true_sku,
            'opt_pred': opt_pred,
            'opt_conf': opt_conf
        })
    # Compute stats
    opt_correct = sum(1 for r in results if r['opt_pred'] == r['true_sku'])
    print(
        f"Optimized Model Accuracy: {opt_correct}/{len(results)} ({100*opt_correct/len(results):.1f}%)")
    # Show a few mismatches
    mismatches = [r for r in results if r['opt_pred'] != r['true_sku']]
    print(f"\nSample mismatches (up to 10):")
    for r in mismatches[:10]:
        print(
            f"VIN: {r['vin']} | Desc: {r['desc']} | True: {r['true_sku']} | Opt: {r['opt_pred']} ({r['opt_conf']:.2f})")


if __name__ == '__main__':
    main()
