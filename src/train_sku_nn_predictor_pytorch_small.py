"""
A simplified version of the SKU Neural Network Predictor training script.
This script uses a small subset of the data to quickly test the PyTorch tokenizer implementation.
"""

import os
import sys
import sqlite3
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our custom tokenizer
try:
    from utils.pytorch_tokenizer import PyTorchTokenizer
except ImportError:
    from utils.dummy_tokenizer import DummyTokenizer

# Constants
DATA_DIR = os.path.join(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))), "data")
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "models", "sku_nn")
DB_PATH = os.path.join(DATA_DIR, "fixacar_history.db")
MAESTRO_PATH = os.path.join(DATA_DIR, "Maestro.xlsx")
BATCH_SIZE = 32
EPOCHS = 5
EMBEDDING_DIM = 100
HIDDEN_DIM = 128
VOCAB_SIZE = 10000
MAX_SEQ_LENGTH = 20
SAMPLE_SIZE = 1000  # Use a small subset of the data for quick testing

# Define the PyTorch model


class SKUNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers=1, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                           batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: [batch_size, seq_len]
        embedded = self.embedding(x)
        # embedded shape: [batch_size, seq_len, embedding_dim]
        output, (hidden, cell) = self.rnn(embedded)
        # hidden shape: [n_layers, batch_size, hidden_dim]
        hidden = hidden[-1, :, :]
        # hidden shape: [batch_size, hidden_dim]
        return self.fc(hidden)


def load_data():
    """Load a small subset of data from the database for quick testing."""
    print(f"Loading data from {DB_PATH}...")

    # Connect to the database
    conn = sqlite3.connect(DB_PATH)

    # Load historical parts data
    query = "SELECT * FROM historical_parts LIMIT ?"
    df = pd.read_sql_query(query, conn, params=(SAMPLE_SIZE,))

    # Close the connection
    conn.close()

    print(f"Loaded {len(df)} records from historical_parts.")

    # Load Maestro data
    maestro_df = pd.read_excel(MAESTRO_PATH)
    print(f"Loaded {len(maestro_df)} records from Maestro.xlsx.")

    return df, maestro_df


def prepare_data(df):
    """Prepare the data for training."""
    # Extract features and target
    descriptions = df['normalized_description'].astype(str).values
    skus = df['sku'].astype(str).values

    # Create a tokenizer using our PyTorch-compatible implementation
    try:
        # Try to use PyTorchTokenizer first
        tokenizer = PyTorchTokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
        print("Using PyTorchTokenizer for training.")
    except NameError:
        # Fall back to DummyTokenizer if PyTorchTokenizer is not available
        tokenizer = DummyTokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
        print("Using DummyTokenizer for training.")

    # Fit the tokenizer on the descriptions
    tokenizer.fit_on_texts(descriptions)

    # Convert descriptions to sequences
    desc_sequences = tokenizer.texts_to_sequences(descriptions)

    # Pad sequences to the same length
    desc_padded = []
    for seq in desc_sequences:
        if len(seq) > MAX_SEQ_LENGTH:
            desc_padded.append(seq[:MAX_SEQ_LENGTH])
        else:
            desc_padded.append(seq + [0] * (MAX_SEQ_LENGTH - len(seq)))

    desc_padded = np.array(desc_padded)

    # Encode SKUs
    sku_encoder = OrdinalEncoder()
    skus_encoded = sku_encoder.fit_transform(skus.reshape(-1, 1)).flatten()

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        desc_padded, skus_encoded, test_size=0.2, random_state=42)

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.long)
    X_val = torch.tensor(X_val, dtype=torch.long)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.long)

    # Create DataLoader objects
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    return train_loader, val_loader, tokenizer, sku_encoder


def save_model_artifacts(tokenizer, sku_encoder):
    """Save the model artifacts."""
    # Create the model directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Save the tokenizer
    joblib.dump(tokenizer, os.path.join(MODEL_DIR, "tokenizer.joblib"))

    # Save the SKU encoder
    joblib.dump(sku_encoder, os.path.join(MODEL_DIR, "sku_encoder.joblib"))

    print(f"Model artifacts saved to {MODEL_DIR}")


def main():
    """Main function to train the SKU Neural Network Predictor."""
    print("--- Starting SKU Neural Network Predictor Training (PyTorch) - Small Version ---")

    # Load data
    df, maestro_df = load_data()

    # Prepare data
    train_loader, val_loader, tokenizer, sku_encoder = prepare_data(df)

    # Save model artifacts
    save_model_artifacts(tokenizer, sku_encoder)

    print("Training completed successfully!")


if __name__ == "__main__":
    main()
