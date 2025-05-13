"""
Optimized PyTorch training script for SKU Neural Network Predictor.
This version includes performance optimizations for faster training.
"""

import pandas as pd
import numpy as np
import os
import sqlite3
import joblib
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
MAX_SEQUENCE_LENGTH = 30  # Reduced from 50 to 30
EMBEDDING_DIM = 64  # Reduced from 100 to 64
HIDDEN_DIM = 64  # Reduced from 128 to 64
BATCH_SIZE = 64  # Increased from 32 to 64
EPOCHS = 50
LEARNING_RATE = 0.001
VIN_BATCH_SIZE = 1000  # Process VINs in batches
# Set to a number (e.g., 50000) to use a subset of data, or None for all data
SAMPLE_SIZE = None  # Using full dataset for training

# --- Load VIN Predictor Models ---
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


def predict_vin_details_batch(vins):
    """Process VIN predictions in batches for better performance."""
    if not model_vin_maker or not model_vin_year or not model_vin_series:
        return [{"Make": "N/A", "Model Year": "N/A", "Series": "N/A"} for _ in vins]

    results = []
    for vin in vins:
        features = extract_vin_features(vin)
        if not features:
            results.append(
                {"Make": "N/A", "Model Year": "N/A", "Series": "N/A"})
            continue

        details = {"Make": "N/A", "Model Year": "N/A", "Series": "N/A"}
        try:
            # Predict Make
            wmi_encoded = encoder_x_vin_maker.transform(
                np.array([[features['wmi']]]))
            if -1 not in wmi_encoded:
                pred_encoded = model_vin_maker.predict(wmi_encoded)
                if pred_encoded[0] != -1:
                    details['Make'] = encoder_y_vin_maker.inverse_transform(
                        pred_encoded.reshape(-1, 1))[0]

            # Predict Model Year
            year_code_encoded = encoder_x_vin_year.transform(
                np.array([[features['year_code']]]))
            if -1 not in year_code_encoded:
                pred_encoded = model_vin_year.predict(year_code_encoded)
                if pred_encoded[0] != -1:
                    details['Model Year'] = encoder_y_vin_year.inverse_transform(
                        pred_encoded.reshape(-1, 1))[0]

            # Predict Series
            series_features_encoded = encoder_x_vin_series.transform(
                np.array([[features['wmi'], features['vds_full']]]))
            if -1 not in series_features_encoded[0]:
                pred_encoded = model_vin_series.predict(
                    series_features_encoded)
                if pred_encoded[0] != -1:
                    details['Series'] = encoder_y_vin_series.inverse_transform(
                        pred_encoded.reshape(-1, 1))[0]
        except Exception as e:
            print(f"Warning: Error predicting details for VIN {vin}: {e}")

        results.append({k: str(v.item()) if isinstance(
            v, np.ndarray) else str(v) for k, v in details.items()})

    return results


def load_and_preprocess_data():
    """Loads data from DB, preprocesses for NN training with optimizations."""
    if not model_vin_maker:  # Check if VIN models loaded
        print("VIN detail prediction models are not available. Cannot preprocess data.")
        return None

    start_time = time.time()
    print(f"Loading data from {DB_PATH}...")

    if not os.path.exists(DB_PATH):
        print(f"Error: Database file not found at {DB_PATH}")
        return None

    conn = sqlite3.connect(DB_PATH)

    # Optionally limit the data size for faster testing
    if SAMPLE_SIZE:
        query = f"SELECT vin_number, normalized_description, sku FROM historical_parts WHERE sku IS NOT NULL LIMIT {SAMPLE_SIZE}"
    else:
        query = "SELECT vin_number, normalized_description, sku FROM historical_parts WHERE sku IS NOT NULL"

    df_history = pd.read_sql_query(query, conn)
    conn.close()

    print(
        f"Loaded {len(df_history)} records from historical_parts in {time.time() - start_time:.2f} seconds.")

    # Process Maestro data if available
    if os.path.exists(MAESTRO_PATH):
        try:
            df_maestro = pd.read_excel(MAESTRO_PATH)
            print(
                f"Loaded {len(df_maestro)} records from Maestro.xlsx (integration pending).")
        except Exception as e:
            print(f"Warning: Could not load or process Maestro.xlsx: {e}")

    df = df_history.copy()
    if df.empty:
        print("No data available after loading.")
        return None

    # Process VIN details in batches with progress tracking
    print("Predicting VIN details for historical data (in batches)...")
    vin_details_list = []
    total_vins = len(df)
    total_batches = (total_vins + VIN_BATCH_SIZE - 1) // VIN_BATCH_SIZE

    # Progress tracking variables
    start_vin_time = time.time()
    last_update_time = start_vin_time
    update_interval = 10  # seconds

    for batch_idx, i in enumerate(range(0, total_vins, VIN_BATCH_SIZE)):
        batch_end = min(i + VIN_BATCH_SIZE, total_vins)

        # Calculate and display progress
        progress_pct = (batch_idx + 1) / total_batches * 100
        current_time = time.time()
        elapsed_time = current_time - start_vin_time

        # Only update progress at intervals to avoid console spam
        if current_time - last_update_time >= update_interval or batch_idx == 0 or batch_idx == total_batches - 1:
            if batch_idx > 0:
                # Estimate remaining time
                time_per_batch = elapsed_time / (batch_idx + 1)
                remaining_batches = total_batches - (batch_idx + 1)
                remaining_time = time_per_batch * remaining_batches

                # Convert to hours, minutes, seconds
                r_hours, r_remainder = divmod(remaining_time, 3600)
                r_minutes, r_seconds = divmod(r_remainder, 60)

                # Format the time string
                if r_hours > 0:
                    time_str = f"{int(r_hours)}h {int(r_minutes)}m remaining"
                elif r_minutes > 0:
                    time_str = f"{int(r_minutes)}m {int(r_seconds)}s remaining"
                else:
                    time_str = f"{int(r_seconds)}s remaining"

                print(
                    f"  Processing VINs {i+1}-{batch_end} of {total_vins} ({progress_pct:.1f}%) - {time_str}")
            else:
                print(
                    f"  Processing VINs {i+1}-{batch_end} of {total_vins} ({progress_pct:.1f}%)")

            last_update_time = current_time

        # Process the batch
        batch_vins = df['vin_number'].iloc[i:batch_end].tolist()
        batch_results = predict_vin_details_batch(batch_vins)
        vin_details_list.extend(batch_results)

    # Print total time for VIN processing
    total_vin_time = time.time() - start_vin_time
    v_hours, v_remainder = divmod(total_vin_time, 3600)
    v_minutes, v_seconds = divmod(v_remainder, 60)

    if v_hours > 0:
        print(
            f"VIN processing completed in {int(v_hours)}h {int(v_minutes)}m {v_seconds:.1f}s")
    elif v_minutes > 0:
        print(
            f"VIN processing completed in {int(v_minutes)}m {v_seconds:.1f}s")
    else:
        print(f"VIN processing completed in {v_seconds:.1f}s")

    df_vin_details = pd.DataFrame(vin_details_list)
    df = pd.concat([df.reset_index(drop=True),
                   df_vin_details.reset_index(drop=True)], axis=1)

    # Clean data
    df.dropna(subset=['Make', 'Model Year', 'Series',
              'normalized_description', 'sku'], inplace=True)
    df = df[(df['Make'] != 'N/A') & (df['Model Year']
                                     != 'N/A') & (df['Series'] != 'N/A')]
    print(f"Records after VIN prediction and NA drop: {len(df)}")

    if df.empty:
        print("No data remaining after VIN detail prediction and cleaning.")
        return None

    # Filter rare SKUs
    sku_counts = df['sku'].value_counts()
    common_skus = sku_counts[sku_counts >= MIN_SKU_FREQUENCY].index
    df = df[df['sku'].isin(common_skus)]
    print(
        f"Records after filtering rare SKUs (min_freq={MIN_SKU_FREQUENCY}): {len(df)}")

    if df.empty or len(df) < 50:
        print("Insufficient data after filtering. Cannot train SKU predictor.")
        return None

    # Encode categorical features
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

    # Save encoders
    os.makedirs(SKU_NN_MODEL_DIR, exist_ok=True)
    for name, encoder in encoders.items():
        joblib.dump(encoder, os.path.join(
            SKU_NN_MODEL_DIR, f'encoder_{name}.joblib'))

    # Tokenize and pad text descriptions
    print("Tokenizing and padding 'normalized_description'...")
    descriptions = df['normalized_description'].astype(str).tolist()

    # Create tokenizer
    try:
        tokenizer = PyTorchTokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
        print("Using PyTorchTokenizer for training.")
    except NameError:
        tokenizer = DummyTokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
        print("Using DummyTokenizer for training.")

    # Fit tokenizer and convert to sequences
    tokenizer.fit_on_texts(descriptions)
    print(f"Found {len(tokenizer.word_index)} unique tokens.")

    sequences = tokenizer.texts_to_sequences(descriptions)

    # Optimized padding
    padded_sequences = tokenizer.pad_sequences(
        sequences,
        maxlen=MAX_SEQUENCE_LENGTH,
        padding='post',
        truncating='post'
    )

    print(f"Padded sequences shape: {padded_sequences.shape}")

    # Save tokenizer
    joblib.dump(tokenizer, os.path.join(SKU_NN_MODEL_DIR, 'tokenizer.joblib'))

    # Prepare features
    X_cat = df[[col + '_encoded' for col in categorical_features]].values
    X_text = padded_sequences
    y_encoded = df['sku_encoded'].values

    print(
        f"Data preprocessing completed in {time.time() - start_time:.2f} seconds.")
    return X_cat, X_text, y_encoded, encoders, tokenizer, num_classes, len(tokenizer.word_index) + 1


# --- Optimized PyTorch Neural Network Model ---
class OptimizedSKUNNModel(nn.Module):
    def __init__(self, cat_input_size, vocab_size, embedding_dim, hidden_size, num_classes, dropout_rate=0.3):
        super(OptimizedSKUNNModel, self).__init__()

        # Embedding layer with dropout
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embed_dropout = nn.Dropout(dropout_rate)

        # Bidirectional LSTM for better feature extraction
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if dropout_rate > 0 else 0
        )

        # Attention mechanism
        self.attention = nn.Linear(hidden_size * 2, 1)

        # Dense layers with batch normalization
        self.batch_norm1 = nn.BatchNorm1d(hidden_size * 2 + cat_input_size)
        self.fc1 = nn.Linear(hidden_size * 2 + cat_input_size, hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def attention_net(self, lstm_output):
        """Apply attention mechanism to LSTM output."""
        # lstm_output shape: (batch_size, seq_len, hidden_size*2)
        attn_weights = torch.tanh(self.attention(
            lstm_output))  # (batch_size, seq_len, 1)
        # (batch_size, seq_len, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)

        # Apply attention weights to LSTM output
        # (batch_size, hidden_size*2)
        context = torch.sum(attn_weights * lstm_output, dim=1)
        return context

    def forward(self, cat_input, text_input):
        # Process text through embedding
        # (batch_size, seq_len, embedding_dim)
        embedded = self.embedding(text_input)
        embedded = self.embed_dropout(embedded)

        # Process through LSTM
        # (batch_size, seq_len, hidden_size*2)
        lstm_out, _ = self.lstm(embedded)

        # Apply attention
        attn_out = self.attention_net(lstm_out)  # (batch_size, hidden_size*2)

        # Concatenate with categorical features
        combined = torch.cat((cat_input, attn_out), dim=1)

        # Process through dense layers
        combined = self.batch_norm1(combined)
        x = torch.relu(self.fc1(combined))
        x = self.dropout1(x)

        x = self.batch_norm2(x)
        logits = self.fc2(x)

        return logits


# --- Main Execution ---
if __name__ == "__main__":
    if not model_vin_maker:  # Critical dependency check
        print("Exiting: VIN detail prediction models failed to load.")
    else:
        print("--- Starting Optimized SKU Neural Network Predictor Training (PyTorch) ---")
        start_time = time.time()

        # Load and preprocess data
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
            print(f"Tokenizer vocabulary size: {vocab_size}")

            # Convert to PyTorch tensors
            X_cat_train_tensor = torch.tensor(X_cat_train, dtype=torch.float32)
            X_text_train_tensor = torch.tensor(X_text_train, dtype=torch.long)
            y_train_tensor = torch.tensor(y_train, dtype=torch.long)

            X_cat_val_tensor = torch.tensor(X_cat_val, dtype=torch.float32)
            X_text_val_tensor = torch.tensor(X_text_val, dtype=torch.long)
            y_val_tensor = torch.tensor(y_val, dtype=torch.long)

            # Create DataLoader with optimized batch size
            train_dataset = TensorDataset(
                X_cat_train_tensor, X_text_train_tensor, y_train_tensor)
            val_dataset = TensorDataset(
                X_cat_val_tensor, X_text_val_tensor, y_val_tensor)

            train_loader = DataLoader(
                train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
            val_loader = DataLoader(
                val_dataset, batch_size=BATCH_SIZE, num_workers=0)

            # Set up device (CPU/GPU)
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {device}")

            # Create optimized model
            model = OptimizedSKUNNModel(
                cat_input_size=X_cat_train.shape[1],
                vocab_size=vocab_size,
                embedding_dim=EMBEDDING_DIM,
                hidden_size=HIDDEN_DIM,
                num_classes=num_classes
            ).to(device)

            # Define loss function and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(
                model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

            # Learning rate scheduler
            scheduler = ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5)

            # Training loop with early stopping
            print("\n--- Training the Optimized SKU NN Model ---")
            print(
                f"Training on {len(train_loader.dataset)} samples, validating on {len(val_loader.dataset)} samples")
            print(
                f"Using device: {device}, batch size: {BATCH_SIZE}, max epochs: {EPOCHS}")

            best_val_loss = float('inf')
            patience = 10
            patience_counter = 0
            training_start_time = time.time()

            for epoch in range(EPOCHS):
                epoch_start_time = time.time()

                # Training phase
                model.train()
                train_loss = 0.0
                correct = 0
                total = 0

                for cat_batch, text_batch, labels_batch in train_loader:
                    cat_batch = cat_batch.to(device)
                    text_batch = text_batch.to(device)
                    labels_batch = labels_batch.to(device)

                    # Zero gradients
                    optimizer.zero_grad()

                    # Forward pass
                    outputs = model(cat_batch, text_batch)
                    loss = criterion(outputs, labels_batch)

                    # Backward pass with gradient clipping
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=1.0)
                    optimizer.step()

                    # Track metrics
                    train_loss += loss.item() * cat_batch.size(0)
                    _, predicted = torch.max(outputs, 1)
                    total += labels_batch.size(0)
                    correct += (predicted == labels_batch).sum().item()

                train_loss = train_loss / len(train_loader.dataset)
                train_acc = correct / total

                # Validation phase
                model.eval()
                val_loss = 0.0
                correct = 0
                total = 0

                with torch.no_grad():
                    for cat_batch, text_batch, labels_batch in val_loader:
                        cat_batch = cat_batch.to(device)
                        text_batch = text_batch.to(device)
                        labels_batch = labels_batch.to(device)

                        outputs = model(cat_batch, text_batch)
                        loss = criterion(outputs, labels_batch)

                        val_loss += loss.item() * cat_batch.size(0)
                        _, predicted = torch.max(outputs, 1)
                        total += labels_batch.size(0)
                        correct += (predicted == labels_batch).sum().item()

                val_loss = val_loss / len(val_loader.dataset)
                val_acc = correct / total

                # Update learning rate
                scheduler.step(val_loss)

                epoch_time = time.time() - epoch_start_time
                print(f"Epoch {epoch+1}/{EPOCHS} [{epoch_time:.1f}s], "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save the best model with timestamp
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    model_filename = f'sku_nn_model_pytorch_optimized_{timestamp}.pth'
                    model_path = os.path.join(SKU_NN_MODEL_DIR, model_filename)

                    # Also save as the default model name for application use
                    default_model_path = os.path.join(
                        SKU_NN_MODEL_DIR, 'sku_nn_model_pytorch_optimized.pth')

                    # Save both versions
                    torch.save(model.state_dict(), model_path)
                    torch.save(model.state_dict(), default_model_path)
                    print(f"  Saved best model to {model_path}")
                    print(
                        f"  Also saved as default model: {default_model_path}")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(
                            f"Early stopping triggered after {epoch+1} epochs")
                        break

            total_time = time.time() - start_time
            hours, remainder = divmod(total_time, 3600)
            minutes, seconds = divmod(remainder, 60)

            print(f"\n--- SKU NN Model Training Complete (PyTorch) ---")
            print(
                f"Total training time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
            print(f"Best model saved to {model_path}")
            print(f"Default model path: {default_model_path}")

        else:
            print("\n--- SKU NN Data Preparation Failed, Model Training Skipped ---")
