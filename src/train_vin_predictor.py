import os
import sys
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
def get_base_path():
    """Get the base path for the application, works for both script and executable."""
    if getattr(sys, 'frozen', False):
        # Running as executable
        return os.path.dirname(sys.executable)
    else:
        # Running as script
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASE_PATH = get_base_path()
CONSOLIDADO_DB_PATH = os.path.join(BASE_PATH, "Source_Files", "processed_consolidado.db")
MODEL_OUTPUT_DIR = os.path.join(BASE_PATH, "models")
# Define minimum frequency for a category to be considered (helps with rare makes/series)
MIN_CATEGORY_FREQUENCY = 5

# --- Feature Extraction ---


def validate_vin_check_digit(vin_str):
    """
    Validate VIN check digit (position 9) using the standard algorithm.
    This is optional validation - some VINs may have incorrect check digits
    but still be valid for maker/model/series extraction.
    """
    # VIN character values for check digit calculation
    char_values = {
        'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8,
        'J': 1, 'K': 2, 'L': 3, 'M': 4, 'N': 5, 'P': 7, 'R': 9, 'S': 2,
        'T': 3, 'U': 4, 'V': 5, 'W': 6, 'X': 7, 'Y': 8, 'Z': 9,
        '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9
    }

    # Position weights for check digit calculation
    weights = [8, 7, 6, 5, 4, 3, 2, 10, 0, 9, 8, 7, 6, 5, 4, 3, 2]

    try:
        total = sum(char_values[char] * weight for char, weight in zip(vin_str, weights))
        calculated_check = total % 11
        expected_check = 'X' if calculated_check == 10 else str(calculated_check)
        return vin_str[8] == expected_check
    except (KeyError, IndexError):
        return False


def clean_vin_for_training(vin):
    """
    Enhanced VIN cleaning specifically for VIN predictor training.
    This function ONLY affects VIN predictor training, not SKU prediction.

    Performs comprehensive VIN validation including:
    - Basic format validation (length, characters)
    - Suspicious pattern detection
    - Structural validation (check digit, year code)
    - Optional: Check digit algorithm validation
    """
    if not vin:
        return None

    # Convert to string and basic cleaning
    vin_str = str(vin).upper().strip()

    # Length validation
    if len(vin_str) != 17:
        return None

    # Character validation (alphanumeric, excluding I, O, Q per VIN standard)
    if not re.match("^[A-HJ-NPR-Z0-9]{17}$", vin_str):
        return None

    # Additional quality checks for VIN predictor training
    # Reject VINs with suspicious patterns that indicate data corruption
    suspicious_patterns = [
        r'^0+',  # All zeros at start
        r'^1+',  # All ones (like '1llllllllllllllll')
        r'(.)\1{10,}',  # Same character repeated 10+ times
        r'^[0-9]{17}$',  # All numeric (unusual for real VINs)
        r'^[A-Z]{17}$',  # All letters (unusual for real VINs)
        r'.*[IOQ].*',  # Contains I, O, or Q (double-check our regex)
        r'^X{5,}',  # Multiple X's (often placeholder data)
        r'^[0]{5,}',  # Multiple zeros (often placeholder data)
        r'TEST|DEMO|SAMPLE',  # Test/demo data
    ]

    for pattern in suspicious_patterns:
        if re.search(pattern, vin_str):
            return None

    # Additional structural validation
    # Position 9 should be a check digit (0-9 or X)
    if vin_str[8] not in '0123456789X':
        return None

    # Position 10 (year code) should be valid year character
    valid_year_codes = 'ABCDEFGHJKLMNPRSTUVWXYZ123456789'
    if vin_str[9] not in valid_year_codes:
        return None

    # Optional: Validate check digit (commented out as it may be too strict)
    # Many VINs in databases have incorrect check digits but valid maker/model/series
    # Uncomment the next 3 lines for stricter validation:
    # if not validate_vin_check_digit(vin_str):
    #     return None

    return vin_str


def clean_vin_for_production(vin):
    """
    Lenient VIN cleaning for production use.
    Only performs basic validation without strict training-specific checks.
    """
    if not vin:
        return None

    # Convert to string and basic cleaning
    vin_str = str(vin).upper().strip()

    # Length validation
    if len(vin_str) != 17:
        return None

    # Character validation (alphanumeric, excluding I, O, Q per VIN standard)
    if not re.match("^[A-HJ-NPR-Z0-9]{17}$", vin_str):
        return None

    return vin_str


def extract_vin_features(vin):
    """Extracts features from a VIN string."""
    # First clean the VIN for training
    cleaned_vin = clean_vin_for_training(vin)
    if not cleaned_vin:
        return None

    features = {
        'wmi': cleaned_vin[0:3],
        'vds': cleaned_vin[3:8],  # Positions 4-8
        # 'check_digit': cleaned_vin[8], # Position 9 - Usually not predictive for Make/Model/Year
        'year_code': cleaned_vin[9],  # Position 10
        'plant_code': cleaned_vin[10],  # Position 11
        # 'sequence': cleaned_vin[11:17] # Positions 12-17 - Usually not predictive
        # Positions 4-9 (including check digit sometimes used)
        'vds_full': cleaned_vin[3:9]
    }
    return features


def extract_vin_features_production(vin):
    """
    Extracts features from a VIN string for production use.
    Uses lenient validation suitable for real-world VIN prediction.
    """
    # Use lenient cleaning for production
    cleaned_vin = clean_vin_for_production(vin)
    if not cleaned_vin:
        return None

    features = {
        'wmi': cleaned_vin[0:3],
        'vds': cleaned_vin[3:8],  # Positions 4-8
        # 'check_digit': cleaned_vin[8], # Position 9 - Usually not predictive for Make/Model/Year
        'year_code': cleaned_vin[9],  # Position 10
        'plant_code': cleaned_vin[10],  # Position 11
        # 'sequence': cleaned_vin[11:17] # Positions 12-17 - Usually not predictive
        # Positions 4-9 (including check digit sometimes used)
        'vds_full': cleaned_vin[3:9]
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
        SELECT vin_number, maker as maker, model as model, series as series
        FROM processed_consolidado
        WHERE vin_number IS NOT NULL
          AND maker IS NOT NULL
          AND model IS NOT NULL
          AND series IS NOT NULL
        """

        df_raw = pd.read_sql_query(query, conn)
        conn.close()

        print(f"Retrieved {len(df_raw)} records from database.")

        # Process each record to extract VIN features with detailed tracking
        valid_vins = 0
        invalid_vins = 0
        filtering_stats = {
            'empty_null': 0,
            'wrong_length': 0,
            'invalid_chars': 0,
            'ioq_chars': 0,  # Track I, O, Q characters specifically
            'suspicious_patterns': 0,
            'invalid_check_digit_pos': 0,
            'invalid_year_code': 0
        }

        for _, row in df_raw.iterrows():
            vin = row['vin_number']

            # Track specific filtering reasons
            if not vin:
                filtering_stats['empty_null'] += 1
                invalid_vins += 1
                continue

            vin_str = str(vin).upper().strip()
            if len(vin_str) != 17:
                filtering_stats['wrong_length'] += 1
                invalid_vins += 1
                continue

            if not re.match("^[A-HJ-NPR-Z0-9]{17}$", vin_str):
                filtering_stats['invalid_chars'] += 1
                invalid_vins += 1
                continue

            features = extract_vin_features(vin)
            if features:
                valid_vins += 1
                # Use model as the target year
                year_target = row['model']

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
                    'model': str(year_target)
                })
            else:
                # This catches additional filtering from extract_vin_features
                filtering_stats['suspicious_patterns'] += 1
                invalid_vins += 1

    except Exception as e:
        print(f"Error processing database: {e}")

    # Report detailed VIN cleaning results
    total_vins = valid_vins + invalid_vins
    if total_vins > 0:
        improvement_pct = (valid_vins / total_vins) * 100
        print(f"\nDetailed VIN Cleaning Results:")
        print(f"  Overall: {valid_vins:,} valid ({improvement_pct:.1f}%) | {invalid_vins:,} filtered ({100-improvement_pct:.1f}%)")
        print(f"  Filtering Breakdown:")
        print(f"    - Empty/Null VINs: {filtering_stats['empty_null']:,}")
        print(f"    - Wrong Length: {filtering_stats['wrong_length']:,}")
        print(f"    - Invalid Characters (I,O,Q): {filtering_stats['invalid_chars']:,}")
        print(f"    - Suspicious Patterns: {filtering_stats['suspicious_patterns']:,}")
        print(f"    - Invalid Check Digit Position: {filtering_stats['invalid_check_digit_pos']:,}")
        print(f"    - Invalid Year Code: {filtering_stats['invalid_year_code']:,}")

        # Calculate data quality improvement
        if invalid_vins > 0:
            print(f"  Data Quality Improvement: +{invalid_vins:,} corrupted VINs removed from training")

    if not all_data:
        print("Error: No valid data loaded for training.")
        return None

    df = pd.DataFrame(all_data)
    print(f"Loaded {len(df)} valid records for VIN predictor training.")

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
        MODEL_OUTPUT_DIR, 'makerr_model.joblib'))
    joblib.dump(encoder_x_maker, os.path.join(
        MODEL_OUTPUT_DIR, 'makerr_encoder_x.joblib'))
    joblib.dump(encoder_y_maker, os.path.join(
        MODEL_OUTPUT_DIR, 'makerr_encoder_y.joblib'))
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
            MODEL_OUTPUT_DIR, 'model_model.joblib'))
        joblib.dump(encoder_x_year, os.path.join(
            MODEL_OUTPUT_DIR, 'model_encoder_x.joblib'))
        joblib.dump(encoder_y_year, os.path.join(
            MODEL_OUTPUT_DIR, 'model_encoder_y.joblib'))
        print("Year model and encoders saved.")

        # Train PyTorch model for Year
        print("Training PyTorch model for Year...")
        num_classes = len(encoder_y_year.categories_[0])
        train_pytorch_model(X_train, y_train, X_test, y_test,
                            input_size=X_train.shape[1],
                            num_classes=num_classes,
                            model_name="model")

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
        MODEL_OUTPUT_DIR, 'series_model.joblib'))
    joblib.dump(encoder_x_series, os.path.join(
        MODEL_OUTPUT_DIR, 'series_encoder_x.joblib'))
    joblib.dump(encoder_y_series, os.path.join(
        MODEL_OUTPUT_DIR, 'series_encoder_y.joblib'))
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
