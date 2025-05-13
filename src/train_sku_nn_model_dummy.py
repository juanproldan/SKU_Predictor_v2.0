from models.sku_nn_pytorch import SKUNNModel
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


def train_pytorch_model(X_cat_train, X_text_train, y_train, X_cat_val, X_text_val, y_val, vocab_size, num_classes):
    """Train a PyTorch model using the provided data."""
    # Create PyTorch datasets
    train_cat_tensor = torch.tensor(X_cat_train, dtype=torch.float32)
    train_text_tensor = torch.tensor(X_text_train, dtype=torch.long)
    train_y_tensor = torch.tensor(y_train, dtype=torch.long)

    val_cat_tensor = torch.tensor(X_cat_val, dtype=torch.float32)
    val_text_tensor = torch.tensor(X_text_val, dtype=torch.long)
    val_y_tensor = torch.tensor(y_val, dtype=torch.long)

    # Create model
    model = SKUNNModel(
        cat_input_size=X_cat_train.shape[1],
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_size=HIDDEN_SIZE,
        num_classes=num_classes
    )

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    print("Training PyTorch model...")
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        outputs = model(train_cat_tensor, train_text_tensor)
        loss = criterion(outputs, train_y_tensor)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_cat_tensor, val_text_tensor)
            val_loss = criterion(val_outputs, val_y_tensor)
            _, predicted = torch.max(val_outputs, 1)
            val_accuracy = (predicted == val_y_tensor).sum(
            ).item() / val_y_tensor.size(0)

        print(
            f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Val Acc: {val_accuracy:.4f}")

    return model


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

    # Train PyTorch model
    model = train_pytorch_model(X_cat_train, X_text_train, y_train,
                                X_cat_val, X_text_val, y_val, vocab_size, num_classes)

    # Save PyTorch model
    print(f"Saving PyTorch model to {PYTORCH_MODEL_PATH}...")
    torch.save(model.state_dict(), PYTORCH_MODEL_PATH)
    print("PyTorch model saved successfully.")


if __name__ == "__main__":
    main()
