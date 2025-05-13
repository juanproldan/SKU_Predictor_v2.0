import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import joblib

# Constants
MAX_SEQUENCE_LENGTH = 50  # Original model sequence length
EMBEDDING_DIM = 100  # Original model embedding dimension
OPTIMIZED_MAX_SEQUENCE_LENGTH = 30  # Optimized model sequence length
OPTIMIZED_EMBEDDING_DIM = 64  # Optimized model embedding dimension
OPTIMIZED_HIDDEN_DIM = 64  # Optimized model hidden dimension


class OptimizedSKUNNModel(nn.Module):
    """
    Optimized PyTorch model for SKU prediction with improved architecture.
    This model includes bidirectional LSTM, attention mechanism, and batch normalization.
    """

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


class SKUNNModel(nn.Module):
    """
    PyTorch model for SKU prediction based on categorical features and text description.
    This model is designed to replace the Keras model in the original implementation.
    """

    def __init__(self, cat_input_size, vocab_size, embedding_dim, hidden_size, num_classes):
        """
        Initialize the model.

        Args:
            cat_input_size: Size of the categorical input features
            vocab_size: Size of the vocabulary for the embedding layer
            embedding_dim: Dimension of the embedding layer
            hidden_size: Size of the LSTM hidden layer
            num_classes: Number of output classes (SKUs)
        """
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
        """
        Forward pass of the model.

        Args:
            cat_input: Categorical input features [batch_size, cat_input_size]
            text_input: Text input (token IDs) [batch_size, max_sequence_length]

        Returns:
            Output logits [batch_size, num_classes]
        """
        # Process text input through embedding and LSTM
        embedded = self.embedding(text_input)
        lstm_out, _ = self.lstm(embedded)
        # Take the output from the last time step
        lstm_out = lstm_out[:, -1, :]

        # Concatenate LSTM output with categorical features
        combined = torch.cat((cat_input, lstm_out), dim=1)

        # Pass through dense layers
        x = F.relu(self.fc1(combined))
        x = self.dropout(x)
        logits = self.fc2(x)

        return logits


def load_model(model_dir):
    """
    Load the PyTorch SKU NN model and its associated encoders.

    Args:
        model_dir: Directory where the model and encoders are stored

    Returns:
        model: Loaded PyTorch model
        encoders: Dictionary of loaded encoders
    """
    # Load encoders
    encoders = {}
    try:
        encoders['Make'] = joblib.load(
            os.path.join(model_dir, 'encoder_Make.joblib'))
        encoders['Model Year'] = joblib.load(
            os.path.join(model_dir, 'encoder_Model Year.joblib'))
        encoders['Series'] = joblib.load(
            os.path.join(model_dir, 'encoder_Series.joblib'))
        encoders['sku'] = joblib.load(
            os.path.join(model_dir, 'encoder_sku.joblib'))

        # Try to load the tokenizer, but use a dummy one if it fails
        try:
            encoders['tokenizer'] = joblib.load(
                os.path.join(model_dir, 'tokenizer.joblib'))
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            # Try to import our PyTorch tokenizer first
            try:
                from utils.pytorch_tokenizer import PyTorchTokenizer
                encoders['tokenizer'] = PyTorchTokenizer(
                    num_words=10000, oov_token="<OOV>")
                print("Using PyTorchTokenizer instead.")
            except ImportError:
                # Fall back to DummyTokenizer if PyTorchTokenizer is not available
                try:
                    from utils.dummy_tokenizer import DummyTokenizer
                    encoders['tokenizer'] = DummyTokenizer(
                        num_words=10000, oov_token="<OOV>")
                    print("Using DummyTokenizer instead.")
                except ImportError:
                    # Last resort: define a minimal tokenizer inline
                    print(
                        "Warning: Could not import tokenizer classes. Using minimal inline tokenizer.")

                    class MinimalTokenizer:
                        def __init__(self):
                            self.word_index = {"<PAD>": 0, "<OOV>": 1}
                            # Add some dummy words
                            for i in range(2, 1000):
                                self.word_index[f"word_{i}"] = i

                        def texts_to_sequences(self, texts):
                            """Convert texts to sequences of integers"""
                            sequences = []
                            for text in texts:
                                words = text.lower().split()
                                seq = []
                                for word in words:
                                    if word in self.word_index:
                                        seq.append(self.word_index[word])
                                    else:
                                        seq.append(self.word_index["<OOV>"])
                                sequences.append(seq)
                            return sequences

                    encoders['tokenizer'] = MinimalTokenizer()
                    print("Using minimal inline tokenizer as last resort.")
    except Exception as e:
        print(f"Error loading encoders: {e}")
        return None, None

    # Try to load optimized model first, then fall back to standard model
    optimized_model_path = os.path.join(
        model_dir, 'sku_nn_model_pytorch_optimized.pth')
    standard_model_path = os.path.join(model_dir, 'sku_nn_model_pytorch.pth')

    # Determine which model file to use
    if os.path.exists(optimized_model_path):
        model_path = optimized_model_path
        use_optimized_model = True
        print("Using optimized model architecture")
    elif os.path.exists(standard_model_path):
        model_path = standard_model_path
        use_optimized_model = False
        print("Using standard model architecture")
    else:
        print(
            f"No model file found at {optimized_model_path} or {standard_model_path}")
        return None, encoders

    try:
        # Get model parameters from encoders
        cat_input_size = 3  # Make, Model Year, Series
        vocab_size = len(encoders['tokenizer'].word_index) + 1
        num_classes = len(encoders['sku'].classes_)

        # Create model instance based on which model file we're using
        if use_optimized_model:
            model = OptimizedSKUNNModel(
                cat_input_size=cat_input_size,
                vocab_size=vocab_size,
                embedding_dim=OPTIMIZED_EMBEDDING_DIM,
                hidden_size=OPTIMIZED_HIDDEN_DIM,
                num_classes=num_classes
            )
        else:
            model = SKUNNModel(
                cat_input_size=cat_input_size,
                vocab_size=vocab_size,
                embedding_dim=EMBEDDING_DIM,
                hidden_size=128,
                num_classes=num_classes
            )

        # Load state dict with special handling for embedding layer size mismatch
        try:
            model.load_state_dict(torch.load(model_path))
        except RuntimeError as e:
            if "embedding.weight" in str(e) and "size mismatch" in str(e):
                print("Handling embedding size mismatch...")
                # Load the state dict
                state_dict = torch.load(model_path)

                # Get the embedding weights from the state dict
                old_embedding = state_dict['embedding.weight']
                old_vocab_size, embedding_dim = old_embedding.shape

                # Initialize the model's embedding with the correct size
                model.embedding = nn.Embedding(old_vocab_size, embedding_dim)

                # Update the state dict with the resized embedding
                state_dict['embedding.weight'] = old_embedding

                # Load the modified state dict
                model.load_state_dict(state_dict)
                print(
                    f"Successfully adjusted embedding layer to size: {old_vocab_size}x{embedding_dim}")
            else:
                # If it's not an embedding size issue, re-raise the exception
                raise

        model.eval()  # Set to evaluation mode

        return model, encoders
    except Exception as e:
        print(f"Error loading PyTorch model: {e}")
        return None, encoders


def predict_sku(model, encoders, make, model_year, series, description, device='cpu'):
    """
    Predict SKU using the PyTorch model.

    Args:
        model: Loaded PyTorch model
        encoders: Dictionary of encoders
        make: Vehicle make
        model_year: Vehicle model year
        series: Vehicle series
        description: Part description
        device: Device to run inference on ('cpu' or 'cuda')

    Returns:
        predicted_sku: Predicted SKU
        confidence: Confidence score (probability)
    """
    if model is None or encoders is None:
        return None, 0.0

    try:
        # Move model to device
        model = model.to(device)

        # Normalize description
        from utils.text_utils import normalize_text
        normalized_desc = normalize_text(description)

        # Encode categorical features with error handling for unseen labels
        try:
            make_enc = encoders['Make'].transform([make.upper()])
        except ValueError:
            print(
                f"Warning: Make '{make}' not seen during training. Using default value.")
            make_enc = np.array([0])  # Use first class as default

        try:
            year_enc = encoders['Model Year'].transform([str(model_year)])
        except ValueError:
            print(
                f"Warning: Model Year '{model_year}' not seen during training. Using default value.")
            year_enc = np.array([0])  # Use first class as default

        try:
            series_enc = encoders['Series'].transform([series.upper()])
        except ValueError:
            print(
                f"Warning: Series '{series}' not seen during training. Using default value.")
            series_enc = np.array([0])  # Use first class as default

        # Tokenize and pad description
        tokenizer = encoders['tokenizer']
        sequences = tokenizer.texts_to_sequences([normalized_desc])

        # Determine which sequence length to use based on model type
        seq_length = OPTIMIZED_MAX_SEQUENCE_LENGTH if isinstance(
            model, OptimizedSKUNNModel) else MAX_SEQUENCE_LENGTH

        # Manual padding (since we're not using Keras pad_sequences)
        padded_seq = [0] * seq_length
        seq = sequences[0]
        if len(seq) > seq_length:
            padded_seq = seq[:seq_length]
        else:
            padded_seq[:len(seq)] = seq

        # Convert inputs to PyTorch tensors
        cat_tensor = torch.tensor(
            [[make_enc[0], year_enc[0], series_enc[0]]], dtype=torch.float32).to(device)
        text_tensor = torch.tensor([padded_seq], dtype=torch.long).to(device)

        # Get prediction
        with torch.no_grad():
            logits = model(cat_tensor, text_tensor)
            probs = F.softmax(logits, dim=1)

        # Get predicted class and confidence
        confidence, predicted_idx = torch.max(probs, dim=1)
        predicted_sku = encoders['sku'].inverse_transform(
            predicted_idx.cpu().numpy())

        return predicted_sku[0], confidence.item()

    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, 0.0
