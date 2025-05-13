import pandas as pd
import numpy as np
import os
import sqlite3
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
try:
    from utils.pytorch_tokenizer import PyTorchTokenizer
except ImportError:
    from utils.dummy_tokenizer import DummyTokenizer

# --- Configuration ---
DB_PATH = "data/fixacar_history.db"
MAESTRO_PATH = "data/Maestro.xlsx"  # Optional, if it contains useful data
VIN_MODEL_DIR = "models"  # Directory for VIN detail predictor models
SKU_NN_MODEL_DIR = "models/sku_nn"  # Directory for SKU NN model and preprocessors
MIN_SKU_FREQUENCY = 3  # Minimum times an SKU must appear to be included in training
VOCAB_SIZE = 10000  # Max number of words to keep in the vocabulary
MAX_SEQUENCE_LENGTH = 50  # Max length of input sequences after padding
EMBEDDING_DIM = 100  # Dimension of the embedding layer (example)
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# --- Load VIN Predictor Models (from train_vin_predictor.py) ---
# This is to ensure consistent feature extraction for VIN details
try:
    from train_vin_predictor import extract_vin_features
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
    print("VIN detail prediction models loaded successfully.")
except Exception as e:
    print(
        f"Error loading VIN detail prediction models: {e}. This script cannot proceed without them.")
    model_vin_maker = None  # Ensure it's None so script can exit if needed


def predict_vin_details_for_training(vin: str) -> dict:
    """Uses loaded VIN models to predict details for training data preparation."""
    if not model_vin_maker or not model_vin_year or not model_vin_series:
        return {"Make": "N/A", "Model Year": "N/A", "Series": "N/A"}

    features = extract_vin_features(vin)
    if not features:
        return {"Make": "N/A", "Model Year": "N/A", "Series": "N/A"}

    details = {"Make": "N/A", "Model Year": "N/A", "Series": "N/A"}
    try:
        wmi_encoded = encoder_x_vin_maker.transform(
            np.array([[features['wmi']]]))
        if -1 not in wmi_encoded:
            pred_encoded = model_vin_maker.predict(wmi_encoded)
            if pred_encoded[0] != -1:
                details['Make'] = encoder_y_vin_maker.inverse_transform(
                    pred_encoded.reshape(-1, 1))[0]

        year_code_encoded = encoder_x_vin_year.transform(
            np.array([[features['year_code']]]))
        if -1 not in year_code_encoded:
            pred_encoded = model_vin_year.predict(year_code_encoded)
            if pred_encoded[0] != -1:
                details['Model Year'] = encoder_y_vin_year.inverse_transform(
                    pred_encoded.reshape(-1, 1))[0]

        series_features_encoded = encoder_x_vin_series.transform(
            np.array([[features['wmi'], features['vds_full']]]))
        if -1 not in series_features_encoded[0]:
            pred_encoded = model_vin_series.predict(series_features_encoded)
            if pred_encoded[0] != -1:
                details['Series'] = encoder_y_vin_series.inverse_transform(
                    pred_encoded.reshape(-1, 1))[0]
    except Exception as e:
        print(
            f"Warning: Error predicting details for VIN {vin} during data prep: {e}")
    return {k: str(v.item()) if isinstance(v, np.ndarray) else str(v) for k, v in details.items()}


# --- Data Loading and Preprocessing ---
def load_and_preprocess_data():
    """Loads data from DB, preprocesses for NN training, including text tokenization."""
    if not model_vin_maker:  # Check if VIN models loaded
        print("VIN detail prediction models are not available. Cannot preprocess data.")
        return None, None, None, None

    print(f"Loading data from {DB_PATH}...")
    if not os.path.exists(DB_PATH):
        print(f"Error: Database file not found at {DB_PATH}")
        return None, None, None, None

    conn = sqlite3.connect(DB_PATH)
    df_history = pd.read_sql_query(
        "SELECT vin_number, normalized_description, sku FROM historical_parts WHERE sku IS NOT NULL", conn)
    conn.close()
    print(f"Loaded {len(df_history)} records from historical_parts.")

    if os.path.exists(MAESTRO_PATH):
        try:
            df_maestro = pd.read_excel(MAESTRO_PATH)
            if not df_maestro.empty and 'Confirmed_SKU' in df_maestro.columns and 'Normalized_Description_Input' in df_maestro.columns:
                print(
                    f"Loaded {len(df_maestro)} records from Maestro.xlsx (integration pending).")
        except Exception as e:
            print(f"Warning: Could not load or process Maestro.xlsx: {e}")

    df = df_history.copy()
    if df.empty:
        print("No data available after loading.")
        return None, None, None, None

    print("Predicting VIN details for historical data...")
    vin_details_list = []
    for i, row in df.iterrows():
        if i % 1000 == 0:
            print(f"  Predicting for VIN {i+1}/{len(df)}...")
        vin_details_list.append(
            predict_vin_details_for_training(row['vin_number']))

    df_vin_details = pd.DataFrame(vin_details_list)
    df = pd.concat([df.reset_index(drop=True),
                   df_vin_details.reset_index(drop=True)], axis=1)
    df.dropna(subset=['Make', 'Model Year', 'Series',
              'normalized_description', 'sku'], inplace=True)
    df = df[(df['Make'] != 'N/A') & (df['Model Year']
                                     != 'N/A') & (df['Series'] != 'N/A')]
    print(f"Records after VIN prediction and NA drop: {len(df)}")

    if df.empty:
        print("No data remaining after VIN detail prediction and cleaning.")
        return None, None, None, None

    sku_counts = df['sku'].value_counts()
    common_skus = sku_counts[sku_counts >= MIN_SKU_FREQUENCY].index
    df = df[df['sku'].isin(common_skus)]
    print(
        f"Records after filtering rare SKUs (min_freq={MIN_SKU_FREQUENCY}): {len(df)}")

    if df.empty or len(df) < 50:
        print("Insufficient data after filtering. Cannot train SKU predictor.")
        return None, None, None, None

    categorical_features = ['Make', 'Model Year', 'Series']
    encoders = {}
    for col in categorical_features:
        le = LabelEncoder()
        df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        print(f"Encoded '{col}', found {len(le.classes_)} unique values.")

    sku_encoder = LabelEncoder()
    df['sku_encoded'] = sku_encoder.fit_transform(df['sku'])
    encoders['sku'] = sku_encoder
    num_classes = len(sku_encoder.classes_)
    print(f"Encoded 'sku', found {num_classes} unique SKUs (classes).")

    os.makedirs(SKU_NN_MODEL_DIR, exist_ok=True)
    for name, encoder in encoders.items():
        joblib.dump(encoder, os.path.join(
            SKU_NN_MODEL_DIR, f'encoder_{name}.joblib'))
    print(f"Saved LabelEncoders to {SKU_NN_MODEL_DIR}")

    # --- Tokenize and Pad Text Descriptions ---
    print("Tokenizing and padding 'normalized_description'...")
    # Ensure descriptions are strings
    descriptions = df['normalized_description'].astype(str).tolist()

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

    print(f"Found {len(tokenizer.word_index)} unique tokens.")

    # Convert text to sequences
    sequences = tokenizer.texts_to_sequences(descriptions)

    # Pad sequences
    padded_sequences = []
    for seq in sequences:
        if len(seq) > MAX_SEQUENCE_LENGTH:
            padded_seq = seq[:MAX_SEQUENCE_LENGTH]
        else:
            padded_seq = seq + [0] * (MAX_SEQUENCE_LENGTH - len(seq))
        padded_sequences.append(padded_seq)

    padded_sequences = np.array(padded_sequences)
    print(f"Padded sequences shape: {padded_sequences.shape}")

    # Save the tokenizer
    tokenizer_path = os.path.join(SKU_NN_MODEL_DIR, 'tokenizer.joblib')
    joblib.dump(tokenizer, tokenizer_path)
    print(f"Saved Tokenizer to {tokenizer_path}")

    # Prepare features for the model
    # X_cat will be the encoded categorical features
    X_cat = df[[col + '_encoded' for col in categorical_features]].values
    # X_text will be the padded sequences
    X_text = padded_sequences
    # y will be the encoded SKUs (integer labels)
    y_encoded = df['sku_encoded'].values

    print("Data loading and preprocessing complete, including text tokenization.")
    # vocab_size for embedding
    return X_cat, X_text, y_encoded, encoders, tokenizer, num_classes, len(tokenizer.word_index)


# --- PyTorch Neural Network Model Definition ---
class SKUNNModel(nn.Module):
    def __init__(self, cat_input_size, vocab_size, embedding_dim, hidden_size, num_classes):
        super(SKUNNModel, self).__init__()

        # Embedding layer for text input
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # LSTM layer for text processing
        self.lstm = nn.LSTM(embedding_dim, hidden_size,
                            batch_first=True, dropout=0.2)

        # Dense layers for classification
        self.fc1 = nn.Linear(hidden_size + cat_input_size, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, cat_input, text_input):
        # Process text input through embedding and LSTM
        embedded = self.embedding(text_input)
        lstm_out, _ = self.lstm(embedded)
        # Take the output from the last time step
        lstm_out = lstm_out[:, -1, :]

        # Concatenate LSTM output with categorical features
        combined = torch.cat((cat_input, lstm_out), dim=1)

        # Pass through dense layers
        x = torch.relu(self.fc1(combined))
        x = self.dropout(x)
        logits = self.fc2(x)

        return logits


# --- Main Execution ---
if __name__ == "__main__":
    if not model_vin_maker:  # Critical dependency check
        print("Exiting: VIN detail prediction models failed to load.")
    else:
        print("--- Starting SKU Neural Network Predictor Training (PyTorch) ---")
        results = load_and_preprocess_data()

        if results is not None:
            X_cat, X_text, y_encoded, encoders, tokenizer, num_classes, vocab_size = results

            # Split data into training and validation sets
            X_cat_train, X_cat_val, X_text_train, X_text_val, y_train, y_val = train_test_split(
                X_cat, X_text, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )

            print(f"\nData preparation successful. Shapes:")
            print(
                f"  X_cat_train: {X_cat_train.shape}, X_text_train: {X_text_train.shape}, y_train: {y_train.shape}")
            print(
                f"  X_cat_val: {X_cat_val.shape}, X_text_val: {X_text_val.shape}, y_val: {y_val.shape}")
            print(f"Number of classes (SKUs): {num_classes}")
            print(
                f"Tokenizer vocabulary size (for embedding layer): {vocab_size}")

            # Convert numpy arrays to PyTorch tensors
            X_cat_train_tensor = torch.tensor(X_cat_train, dtype=torch.float32)
            X_text_train_tensor = torch.tensor(X_text_train, dtype=torch.long)
            y_train_tensor = torch.tensor(y_train, dtype=torch.long)

            X_cat_val_tensor = torch.tensor(X_cat_val, dtype=torch.float32)
            X_text_val_tensor = torch.tensor(X_text_val, dtype=torch.long)
            y_val_tensor = torch.tensor(y_val, dtype=torch.long)

            # Create DataLoader for batching
            train_dataset = TensorDataset(
                X_cat_train_tensor, X_text_train_tensor, y_train_tensor)
            val_dataset = TensorDataset(
                X_cat_val_tensor, X_text_val_tensor, y_val_tensor)

            train_loader = DataLoader(
                train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

            # Create model
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {device}")

            model = SKUNNModel(
                cat_input_size=X_cat_train.shape[1],
                vocab_size=vocab_size,
                embedding_dim=EMBEDDING_DIM,
                hidden_size=128,
                num_classes=num_classes
            ).to(device)

            # Define loss function and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

            # Training loop
            print("\n--- Training the SKU NN Model (PyTorch) ---")
            best_val_loss = float('inf')
            patience = 10
            patience_counter = 0

            for epoch in range(EPOCHS):
                # Training
                model.train()
                train_loss = 0.0
                correct = 0
                total = 0

                for cat_batch, text_batch, labels_batch in train_loader:
                    cat_batch, text_batch, labels_batch = cat_batch.to(
                        device), text_batch.to(device), labels_batch.to(device)

                    # Zero the parameter gradients
                    optimizer.zero_grad()

                    # Forward pass
                    outputs = model(cat_batch, text_batch)
                    loss = criterion(outputs, labels_batch)

                    # Backward pass and optimize
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item() * cat_batch.size(0)
                    _, predicted = torch.max(outputs, 1)
                    total += labels_batch.size(0)
                    correct += (predicted == labels_batch).sum().item()

                train_loss = train_loss / len(train_loader.dataset)
                train_acc = correct / total

                # Validation
                model.eval()
                val_loss = 0.0
                correct = 0
                total = 0

                with torch.no_grad():
                    for cat_batch, text_batch, labels_batch in val_loader:
                        cat_batch, text_batch, labels_batch = cat_batch.to(
                            device), text_batch.to(device), labels_batch.to(device)

                        outputs = model(cat_batch, text_batch)
                        loss = criterion(outputs, labels_batch)

                        val_loss += loss.item() * cat_batch.size(0)
                        _, predicted = torch.max(outputs, 1)
                        total += labels_batch.size(0)
                        correct += (predicted == labels_batch).sum().item()

                val_loss = val_loss / len(val_loader.dataset)
                val_acc = correct / total

                print(f"Epoch {epoch+1}/{EPOCHS}, "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save the best model
                    model_path = os.path.join(
                        SKU_NN_MODEL_DIR, 'sku_nn_model_pytorch.pth')
                    torch.save(model.state_dict(), model_path)
                    print(f"  Saved best model to {model_path}")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(
                            f"Early stopping triggered after {epoch+1} epochs")
                        break

            print(f"\n--- SKU NN Model Training Complete (PyTorch) ---")
            print(
                f"Best model saved to {os.path.join(SKU_NN_MODEL_DIR, 'sku_nn_model_pytorch.pth')}")

        else:
            print("\n--- SKU NN Data Preparation Failed, Model Training Skipped ---")
