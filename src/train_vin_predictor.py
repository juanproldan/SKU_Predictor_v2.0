import os
import pandas as pd
import sqlite3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
import re

# --- Configuration ---
CONSOLIDADO_DB_PATH = "data/consolidado.db"
MODEL_OUTPUT_DIR = "models"
# Define minimum frequency for a category to be considered (helps with rare makes/series)
MIN_CATEGORY_FREQUENCY = 5

# --- Feature Extraction ---


def extract_vin_features(vin):
    """Extracts features from a VIN string."""
    if not isinstance(vin, str) or len(vin) != 17:
        return None

    # Basic validation (alphanumeric, excluding I, O, Q)
    if not re.match("^[A-HJ-NPR-Z0-9]{17}$", vin):
        print(
            f"Warning: VIN '{vin}' contains invalid characters or length. Skipping.")
        return None

    features = {
        'wmi': vin[0:3],
        'vds': vin[3:8],  # Positions 4-8
        # 'check_digit': vin[8], # Position 9 - Usually not predictive for Make/Model/Year
        'year_code': vin[9],  # Position 10
        'plant_code': vin[10],  # Position 11
        # 'sequence': vin[11:17] # Positions 12-17 - Usually not predictive
        # Positions 4-9 (including check digit sometimes used)
        'vds_full': vin[3:9]
    }
    return features


# --- Year Code Mapping ---
# Standard VIN Year Codes (adjust if needed, covers 1980-2039)
VIN_YEAR_MAP = {
    'A': 1980, 'B': 1981, 'C': 1982, 'D': 1983, 'E': 1984, 'F': 1985, 'G': 1986, 'H': 1987,
    'J': 1988, 'K': 1989, 'L': 1990, 'M': 1991, 'N': 1992, 'P': 1993, 'R': 1994, 'S': 1995,
    'T': 1996, 'V': 1997, 'W': 1998, 'X': 1999, 'Y': 2000, '1': 2001, '2': 2002, '3': 2003,
    '4': 2004, '5': 2005, '6': 2006, '7': 2007, '8': 2008, '9': 2009, 'A': 2010, 'B': 2011,
    'C': 2012, 'D': 2013, 'E': 2014, 'F': 2015, 'G': 2016, 'H': 2017, 'J': 2018, 'K': 2019,
    'L': 2020, 'M': 2021, 'N': 2022, 'P': 2023, 'R': 2024, 'S': 2025, 'T': 2026, 'V': 2027,
    'W': 2028, 'X': 2029, 'Y': 2030, '1': 2031, '2': 2032, '3': 2033, '4': 2034, '5': 2035,
    '6': 2036, '7': 2037, '8': 2038, '9': 2039
}
# Note: Year codes repeat. The map prioritizes recent years for overlaps (e.g., 'A' is 2010, not 1980).
# This might need adjustment based on the actual year range in your data.


def decode_year(year_code):
    """Decodes the year from the VIN's 10th character."""
    return VIN_YEAR_MAP.get(year_code)

# --- Data Loading and Preparation ---


def load_and_prepare_data():
    """Loads data from SQLite database, extracts features and targets."""
    all_data = []
    print("Loading historical data for training...")

    if not os.path.exists(CONSOLIDADO_DB_PATH):
        print(f"Error: Database file not found at {CONSOLIDADO_DB_PATH}")
        return None

    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(CONSOLIDADO_DB_PATH)

        # Query the database for relevant data
        query = """
        SELECT vin_number, vin_make as maker, vin_year as fabrication_year, vin_series as series
        FROM filtered_bids
        WHERE vin_number IS NOT NULL
          AND vin_make IS NOT NULL
          AND vin_year IS NOT NULL
          AND vin_series IS NOT NULL
        """

        df_raw = pd.read_sql_query(query, conn)
        conn.close()

        print(f"Retrieved {len(df_raw)} records from database.")

        # Process each record to extract VIN features
        for _, row in df_raw.iterrows():
            vin = row['vin_number']
            features = extract_vin_features(vin)

            if features:
                # Use fabrication_year as the target year
                year_target = row['fabrication_year']

                # Decode year from VIN code for potential consistency check
                year_decoded = decode_year(features['year_code'])

                # Add the processed data
                all_data.append({
                    'wmi': features['wmi'],
                    'vds': features['vds'],
                    'vds_full': features['vds_full'],
                    'year_code': features['year_code'],
                    'plant_code': features['plant_code'],
                    'maker': row['maker'],
                    'series': row['series'],
                    # Target year as string
                    'year': str(year_target)
                })

    except Exception as e:
        print(f"Error processing database: {e}")

    if not all_data:
        print("Error: No valid data loaded for training.")
        return None

    df = pd.DataFrame(all_data)
    print(f"Loaded {len(df)} valid records for training.")

    # --- Data Cleaning: Handle rare categories ---
    print("Cleaning data (handling rare categories)...")
    for col in ['maker', 'series']:
        counts = df[col].value_counts()
        # Keep only categories that appear at least MIN_CATEGORY_FREQUENCY times
        valid_categories = counts[counts >= MIN_CATEGORY_FREQUENCY].index
        df = df[df[col].isin(valid_categories)]
        print(
            f"  Filtered {col}, kept {len(valid_categories)} categories. Records remaining: {len(df)}")

    if len(df) < 50:  # Need a reasonable amount of data to train
        print(
            f"Error: Insufficient data ({len(df)} records) after filtering rare categories. Cannot train effectively.")
        return None

    return df

# --- Model Training ---

# PyTorch model definitions


class SimpleClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def train_pytorch_model(X_train, y_train, X_test, y_test, input_size, num_classes, model_name):
    """Train a PyTorch model and return it along with accuracy."""
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)

    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Initialize model
    model = SimpleClassifier(input_size=input_size,
                             hidden_size=64, num_classes=num_classes)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == y_test_tensor).sum().item() / \
            y_test_tensor.size(0)

    print(f"{model_name} PyTorch Model Accuracy: {accuracy:.4f}")

    # Save the PyTorch model
    torch.save(model.state_dict(), os.path.join(MODEL_OUTPUT_DIR,
               f'vin_{model_name.lower()}_model_pytorch.pth'))

    return model, accuracy


def train_and_save_models(df):
    """Trains models for Maker, Year, Series and saves them."""
    if df is None or df.empty:
        print("No data available for training.")
        return

    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

    # --- 1. Maker Prediction (using WMI) ---
    print("\n--- Training Maker Predictor ---")
    X_maker = df[['wmi']]
    y_maker = df['maker']

    # Encode features and target
    encoder_x_maker = OrdinalEncoder(
        handle_unknown='use_encoded_value', unknown_value=-1)
    encoder_y_maker = OrdinalEncoder(
        handle_unknown='use_encoded_value', unknown_value=-1)

    X_maker_encoded = encoder_x_maker.fit_transform(X_maker)
    y_maker_encoded = encoder_y_maker.fit_transform(
        y_maker.values.reshape(-1, 1)).ravel()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_maker_encoded, y_maker_encoded, test_size=0.2, random_state=42)

    # Train model (CategoricalNB is suitable for discrete features)
    model_maker = CategoricalNB(min_categories=len(
        encoder_x_maker.categories_[0]))  # Needs number of categories
    model_maker.fit(X_train, y_train)

    # Evaluate
    y_pred = model_maker.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Maker Model Accuracy: {accuracy:.4f}")

    # Save model and encoders
    joblib.dump(model_maker, os.path.join(
        MODEL_OUTPUT_DIR, 'vin_maker_model.joblib'))
    joblib.dump(encoder_x_maker, os.path.join(
        MODEL_OUTPUT_DIR, 'vin_maker_encoder_x.joblib'))
    joblib.dump(encoder_y_maker, os.path.join(
        MODEL_OUTPUT_DIR, 'vin_maker_encoder_y.joblib'))
    print("Maker model and encoders saved.")

    # Train PyTorch model for Maker
    print("Training PyTorch model for Maker...")
    num_classes = len(encoder_y_maker.categories_[0])
    train_pytorch_model(X_train, y_train, X_test, y_test,
                        input_size=X_train.shape[1],
                        num_classes=num_classes,
                        model_name="Maker")

    # --- 2. Year Prediction (using Year Code) ---
    # This is mostly a direct mapping, but we can frame it as prediction
    print("\n--- Training Year Predictor ---")
    # Use only records where year code is known and maps to a year
    df_year = df[df['year_code'].isin(VIN_YEAR_MAP.keys())].copy()
    if df_year.empty:
        print("No data with valid year codes found. Skipping Year predictor.")
    else:
        df_year['year_decoded'] = df_year['year_code'].map(
            VIN_YEAR_MAP).astype(str)  # Target is the decoded year string
        X_year = df_year[['year_code']]
        y_year = df_year['year_decoded']  # Use decoded year as target

        encoder_x_year = OrdinalEncoder(
            handle_unknown='use_encoded_value', unknown_value=-1)
        encoder_y_year = OrdinalEncoder(
            handle_unknown='use_encoded_value', unknown_value=-1)

        X_year_encoded = encoder_x_year.fit_transform(X_year)
        y_year_encoded = encoder_y_year.fit_transform(
            y_year.values.reshape(-1, 1)).ravel()

        X_train, X_test, y_train, y_test = train_test_split(
            X_year_encoded, y_year_encoded, test_size=0.2, random_state=42)

        model_year = CategoricalNB(
            min_categories=len(encoder_x_year.categories_[0]))
        model_year.fit(X_train, y_train)

        y_pred = model_year.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        # Should be high if mapping is consistent
        print(f"Year Model Accuracy: {accuracy:.4f}")

        joblib.dump(model_year, os.path.join(
            MODEL_OUTPUT_DIR, 'vin_year_model.joblib'))
        joblib.dump(encoder_x_year, os.path.join(
            MODEL_OUTPUT_DIR, 'vin_year_encoder_x.joblib'))
        joblib.dump(encoder_y_year, os.path.join(
            MODEL_OUTPUT_DIR, 'vin_year_encoder_y.joblib'))
        print("Year model and encoders saved.")

        # Train PyTorch model for Year
        print("Training PyTorch model for Year...")
        num_classes = len(encoder_y_year.categories_[0])
        train_pytorch_model(X_train, y_train, X_test, y_test,
                            input_size=X_train.shape[1],
                            num_classes=num_classes,
                            model_name="Year")

    # --- 3. Series Prediction (using WMI + VDS) ---
    # This is the most complex and likely least accurate part
    print("\n--- Training Series Predictor ---")
    # Features: WMI and VDS
    X_series = df[['wmi', 'vds_full']]  # Using positions 1-9
    y_series = df['series']

    encoder_x_series = OrdinalEncoder(
        handle_unknown='use_encoded_value', unknown_value=-1)
    encoder_y_series = OrdinalEncoder(
        handle_unknown='use_encoded_value', unknown_value=-1)

    X_series_encoded = encoder_x_series.fit_transform(X_series)
    y_series_encoded = encoder_y_series.fit_transform(
        y_series.values.reshape(-1, 1)).ravel()

    X_train, X_test, y_train, y_test = train_test_split(
        X_series_encoded, y_series_encoded, test_size=0.2, random_state=42)

    # Calculate min_categories for each feature
    min_cats_series = [len(cat) for cat in encoder_x_series.categories_]

    model_series = CategoricalNB(min_categories=min_cats_series)
    model_series.fit(X_train, y_train)

    y_pred = model_series.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    # Expect lower accuracy here
    print(f"Series Model Accuracy: {accuracy:.4f}")

    joblib.dump(model_series, os.path.join(
        MODEL_OUTPUT_DIR, 'vin_series_model.joblib'))
    joblib.dump(encoder_x_series, os.path.join(
        MODEL_OUTPUT_DIR, 'vin_series_encoder_x.joblib'))
    joblib.dump(encoder_y_series, os.path.join(
        MODEL_OUTPUT_DIR, 'vin_series_encoder_y.joblib'))
    print("Series model and encoders saved.")

    # Train PyTorch model for Series
    print("Training PyTorch model for Series...")
    num_classes = len(encoder_y_series.categories_[0])
    train_pytorch_model(X_train, y_train, X_test, y_test,
                        input_size=X_train.shape[1],
                        num_classes=num_classes,
                        model_name="Series")


# --- Main Execution ---
if __name__ == "__main__":
    print("--- Starting VIN Predictor Training ---")
    prepared_data = load_and_prepare_data()

    if prepared_data is not None:
        train_and_save_models(prepared_data)
        print("\n--- Training Complete ---")
    else:
        print("\n--- Training Aborted due to data loading/preparation issues ---")
