import os
import sys
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.append('src')

# Configuration
MODEL_DIR = "models"
SKU_NN_MODEL_DIR = os.path.join(MODEL_DIR, "sku_nn")
PYTORCH_MODEL_PATH = os.path.join(SKU_NN_MODEL_DIR, "sku_nn_model_pytorch.pth")
EMBEDDING_DIM = 100
HIDDEN_SIZE = 128
MAX_SEQUENCE_LENGTH = 50
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001


def load_encoders():
    """Load the encoders used by the PyTorch model."""
    encoders = {}
    try:
        encoders['Make'] = joblib.load(os.path.join(
            SKU_NN_MODEL_DIR, 'encoder_Make.joblib'))
        encoders['Model Year'] = joblib.load(os.path.join(
            SKU_NN_MODEL_DIR, 'encoder_Model Year.joblib'))
        encoders['Series'] = joblib.load(os.path.join(
            SKU_NN_MODEL_DIR, 'encoder_Series.joblib'))
        encoders['sku'] = joblib.load(os.path.join(
            SKU_NN_MODEL_DIR, 'encoder_sku.joblib'))
        encoders['tokenizer'] = joblib.load(
            os.path.join(SKU_NN_MODEL_DIR, 'tokenizer.joblib'))
        return encoders
    except Exception as e:
        print(f"Error loading encoders: {e}")
        return None


def generate_dummy_data(encoders, num_samples=1000):
    """Generate dummy data for training the PyTorch model."""
    # Create random categorical features
    make_classes = encoders['Make'].classes_
    year_classes = encoders['Model Year'].classes_
    series_classes = encoders['Series'].classes_

    # Generate random indices for each category
    make_indices = np.random.randint(0, len(make_classes), num_samples)
    year_indices = np.random.randint(0, len(year_classes), num_samples)
    series_indices = np.random.randint(0, len(series_classes), num_samples)

    # Create categorical input array
    X_cat = np.column_stack((make_indices, year_indices, series_indices))

    # Create random text input (token IDs)
    vocab_size = len(encoders['tokenizer'].word_index) + 1
    X_text = np.random.randint(
        0, vocab_size, (num_samples, MAX_SEQUENCE_LENGTH))

    # Create random target labels
    num_classes = len(encoders['sku'].classes_)
    y = np.random.randint(0, num_classes, num_samples)

    # Split into train and validation sets
    X_cat_train, X_cat_val, X_text_train, X_text_val, y_train, y_val = train_test_split(
        X_cat, X_text, y, test_size=0.2, random_state=42
    )

    return X_cat_train, X_cat_val, X_text_train, X_text_val, y_train, y_val, vocab_size, num_classes


def main():
    """Main function to train a PyTorch model with dummy data."""
    print("Starting PyTorch model training with dummy data...")

    # Load encoders
    encoders = load_encoders()
    if not encoders:
        print("Failed to load encoders. Cannot proceed.")
        return

    # Generate dummy data for training
    print("Generating dummy training data...")
    X_cat_train, X_cat_val, X_text_train, X_text_val, y_train, y_val, vocab_size, num_classes = generate_dummy_data(
        encoders)

    print("This script is obsolete for the standard model. Please repurpose or delete.")
    return


if __name__ == "__main__":
    main()
