import tkinter as tk
from tkinter import ttk
from tkinter import messagebox  # For showing error popups
import os
import pandas as pd
# import requests # No longer needed
import sqlite3  # For database connection
from collections import defaultdict  # For counting frequencies
import datetime  # For timestamping Maestro entries
import json  # For loading consolidado
import joblib  # To load trained models
import numpy as np  # For model input reshaping
import re  # For VIN validation
import torch  # For PyTorch
# Import our PyTorch model implementation
try:
    from models.sku_nn_pytorch import load_model, predict_sku
except ImportError:
    # Fallback for direct execution
    from .models.sku_nn_pytorch import load_model, predict_sku
# Assuming text_utils and train_vin_predictor are in the same directory or src is in PYTHONPATH
try:
    from utils.text_utils import normalize_text
    from utils.dummy_tokenizer import DummyTokenizer
    # Import feature extraction and year decoding from the training script
    from train_vin_predictor import extract_vin_features, decode_year
except ImportError:
    # Fallback for direct execution if src is not in path (e.g. during development)
    from .utils.text_utils import normalize_text
    from .utils.dummy_tokenizer import DummyTokenizer
    from .train_vin_predictor import extract_vin_features, decode_year

# --- Determine Project Root Path ---
# Get the directory of the current script (src)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Assume the project root is one level up from 'src'
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


# --- Configuration (using absolute paths) ---
# Assuming Equivalencias.xlsx is at the project root
DEFAULT_EQUIVALENCIAS_PATH = os.path.join(
    PROJECT_ROOT, "Source_Files", "Equivalencias.xlsx")
DEFAULT_MAESTRO_PATH = os.path.join(PROJECT_ROOT, "data", "Maestro.xlsx")
DEFAULT_DB_PATH = os.path.join(PROJECT_ROOT, "data", "fixacar_history.db")
# Directory where models are saved
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
# Define pattern and count for loading VIN details from chunks (Used by load_vin_details_from_chunks)
# This path might also need adjustment if it's not relative to CWD
# Keeping this relative for now, assuming it's handled elsewhere or CWD is intended for these specific chunks
CONSOLIDADO_CHUNK_PATTERN_FOR_VIN_LOAD = "Consolidado_chunk_{}.json"
NUM_CONSOLIDADO_CHUNKS_FOR_VIN_LOAD = 10


# In-memory data stores
equivalencias_map_global = {}
maestro_data_global = []  # This will hold the list of dictionaries
# VIN details lookup is replaced by models

# Loaded Models and Encoders
model_maker = None
encoder_x_maker = None
encoder_y_maker = None
model_year = None
encoder_x_year = None
encoder_y_year = None
model_series = None
encoder_x_series = None
encoder_y_series = None

# SKU NN Model and Encoders/Tokenizer
sku_nn_model = None
sku_nn_encoder_make = None
sku_nn_encoder_model_year = None
sku_nn_encoder_series = None
sku_nn_tokenizer_desc = None  # Assuming description is an input
sku_nn_encoder_sku = None
SKU_NN_MODEL_DIR = os.path.join(MODEL_DIR, "sku_nn")


class FixacarApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fixacar SKU Finder v1.0 (with VIN Predictor)")
        self.root.geometry("800x750")  # Increased height

        # Load initial data and models
        self.load_all_data_and_models()

        # Setup UI
        self.create_widgets()

    def load_all_data_and_models(self):
        """Loads data files and trained prediction models on startup."""
        global equivalencias_map_global, maestro_data_global
        global model_maker, encoder_x_maker, encoder_y_maker
        global model_year, encoder_x_year, encoder_y_year
        global model_series, encoder_x_series, encoder_y_series

        print("--- Loading Application Data & Models ---")
        equivalencias_map_global = self.load_equivalencias_data(
            DEFAULT_EQUIVALENCIAS_PATH)
        maestro_data_global = self.load_maestro_data(
            DEFAULT_MAESTRO_PATH, equivalencias_map_global)

        # Load Models
        print("Loading VIN prediction models...")
        try:
            model_maker = joblib.load(os.path.join(
                MODEL_DIR, 'vin_maker_model.joblib'))
            encoder_x_maker = joblib.load(os.path.join(
                MODEL_DIR, 'vin_maker_encoder_x.joblib'))
            encoder_y_maker = joblib.load(os.path.join(
                MODEL_DIR, 'vin_maker_encoder_y.joblib'))
            print("  Maker model loaded.")

            model_year = joblib.load(os.path.join(
                MODEL_DIR, 'vin_year_model.joblib'))
            encoder_x_year = joblib.load(os.path.join(
                MODEL_DIR, 'vin_year_encoder_x.joblib'))
            encoder_y_year = joblib.load(os.path.join(
                MODEL_DIR, 'vin_year_encoder_y.joblib'))
            print("  Year model loaded.")

            model_series = joblib.load(os.path.join(
                MODEL_DIR, 'vin_series_model.joblib'))
            encoder_x_series = joblib.load(os.path.join(
                MODEL_DIR, 'vin_series_encoder_x.joblib'))
            encoder_y_series = joblib.load(os.path.join(
                MODEL_DIR, 'vin_series_encoder_y.joblib'))
            print("  Series model loaded.")
            print("All models loaded successfully.")

        except FileNotFoundError as e:
            print(
                f"Error loading model file: {e}. VIN prediction will not work.")
            messagebox.showerror(
                "Model Loading Error", f"Could not load model file: {e}\nPlease ensure models are trained and present in the '{MODEL_DIR}' directory.")
            # Set models to None so prediction attempts fail gracefully
            model_maker, model_year, model_series = None, None, None
        except Exception as e:
            print(f"An unexpected error occurred loading models: {e}")
            messagebox.showerror(
                "Model Loading Error", f"An unexpected error occurred loading models: {e}")
            model_maker, model_year, model_series = None, None, None

        # Load SKU NN Model and preprocessors
        global sku_nn_model, sku_nn_encoder_make, sku_nn_encoder_model_year, sku_nn_encoder_series, sku_nn_tokenizer_desc, sku_nn_encoder_sku
        print("Loading SKU NN model and preprocessors...")
        try:
            # Load PyTorch model and encoders
            sku_nn_model_path = os.path.join(
                SKU_NN_MODEL_DIR, 'sku_nn_model_pytorch.pth')

            # First load the encoders separately to ensure they're available even if model loading fails
            sku_nn_encoder_make = joblib.load(os.path.join(
                SKU_NN_MODEL_DIR, 'encoder_Make.joblib'))
            print("  SKU NN Make encoder loaded.")
            sku_nn_encoder_model_year = joblib.load(os.path.join(
                SKU_NN_MODEL_DIR, 'encoder_Model Year.joblib'))
            print("  SKU NN Model Year encoder loaded.")
            sku_nn_encoder_series = joblib.load(os.path.join(
                SKU_NN_MODEL_DIR, 'encoder_Series.joblib'))
            print("  SKU NN Series encoder loaded.")
            try:
                sku_nn_tokenizer_desc = joblib.load(os.path.join(
                    SKU_NN_MODEL_DIR, 'tokenizer.joblib'))
                print("  SKU NN Description tokenizer loaded.")
            except Exception as e:
                print(f"  Error loading tokenizer: {e}")
                # Try to use our PyTorch tokenizer first
                try:
                    from utils.pytorch_tokenizer import PyTorchTokenizer
                    sku_nn_tokenizer_desc = PyTorchTokenizer(
                        num_words=10000, oov_token="<OOV>")
                    print("  Using PyTorchTokenizer instead.")
                except ImportError:
                    # Fall back to DummyTokenizer
                    from utils.dummy_tokenizer import DummyTokenizer
                    sku_nn_tokenizer_desc = DummyTokenizer(
                        num_words=10000, oov_token="<OOV>")
                    print("  Using DummyTokenizer instead.")

            sku_nn_encoder_sku = joblib.load(os.path.join(
                SKU_NN_MODEL_DIR, 'encoder_sku.joblib'))
            print("  SKU NN SKU encoder loaded.")

            # Now try to load the PyTorch model
            if os.path.exists(sku_nn_model_path):
                # Use our custom load_model function from sku_nn_pytorch.py
                sku_nn_model, _ = load_model(SKU_NN_MODEL_DIR)
                if sku_nn_model:
                    print("  SKU NN PyTorch model loaded successfully.")
                else:
                    print("  Failed to load SKU NN PyTorch model.")
            else:
                print(
                    f"  SKU NN PyTorch model file not found at {sku_nn_model_path}")
                print("  Note: You need to train and save a PyTorch model first.")
                sku_nn_model = None

        except FileNotFoundError as e:
            print(
                f"Error loading SKU NN model file: {e}. SKU NN prediction will not work.")
            messagebox.showerror("SKU NN Model Error",
                                 f"Could not load SKU NN model file: {e}")
            sku_nn_model = None  # Ensure it's None if loading fails
        except Exception as e:
            print(f"An unexpected error occurred loading SKU NN models: {e}")
            messagebox.showerror(
                "SKU NN Model Error", f"An unexpected error occurred loading SKU NN models: {e}")
            sku_nn_model = None

        # DB connection will be established when needed for search
        print("--- Application Data & Model Loading Complete ---")

    def load_equivalencias_data(self, file_path: str) -> dict:
        # (Content remains the same as before)
        print(f"Loading equivalencias from: {file_path}")
        if not os.path.exists(file_path):
            print(
                f"Warning: Equivalencias file not found at {file_path}. Equivalency linking will be disabled.")
            return {}
        try:
            df = pd.read_excel(file_path, sheet_name=0)
            equivalencias_map = {}
            for index, row in df.iterrows():
                equivalencia_row_id = index + 1
                for col_name in df.columns:
                    term = row[col_name]
                    if pd.notna(term) and str(term).strip():
                        normalized_term = normalize_text(str(term))
                        if normalized_term:
                            equivalencias_map[normalized_term] = equivalencia_row_id
            print(
                f"Loaded {len(equivalencias_map)} normalized term mappings from {len(df)} rows in Equivalencias.")
            return equivalencias_map
        except Exception as e:
            print(
                f"Error loading or processing Equivalencias.xlsx: {e}. Equivalency linking will be disabled.")
            return {}

    def load_maestro_data(self, file_path: str, equivalencias_map: dict) -> list:
        # (Content remains the same as before)
        print(f"Loading maestro data from: {file_path}")
        maestro_columns = [
            'Maestro_ID', 'VIN_Make', 'VIN_Model', 'VIN_Year_Min', 'VIN_Year_Max',
            'VIN_Series_Trim', 'VIN_BodyStyle', 'Original_Description_Input',
            'Normalized_Description_Input', 'Equivalencia_Row_ID', 'Confirmed_SKU',
            'Confidence', 'Source', 'Date_Added'
        ]
        data_dir = os.path.dirname(file_path)
        if data_dir and not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)
        if not os.path.exists(file_path):
            print(
                f"Maestro file not found at {file_path}. Creating a new one with headers.")
            df_maestro = pd.DataFrame(columns=maestro_columns)
            try:
                df_maestro.to_excel(file_path, index=False)
                return []
            except Exception as e:
                print(
                    f"Error creating Maestro.xlsx: {e}. Maestro data will be empty.")
                return []
        try:
            df_maestro = pd.read_excel(file_path, sheet_name=0)
            if 'Maestro_ID' in df_maestro.columns:
                df_maestro['Maestro_ID'] = pd.to_numeric(
                    df_maestro['Maestro_ID'], errors='coerce').fillna(0).astype(int)
            if df_maestro.empty:
                print("Maestro file is empty.")
                return []
            maestro_list = []
            for _, row in df_maestro.iterrows():
                entry = row.to_dict()
                original_desc = entry.get('Original_Description_Input', "")
                if pd.notna(original_desc):
                    normalized_desc = normalize_text(str(original_desc))
                    entry['Normalized_Description_Input'] = normalized_desc
                    entry['Equivalencia_Row_ID'] = equivalencias_map.get(
                        normalized_desc)
                else:
                    entry['Normalized_Description_Input'] = ""
                    entry['Equivalencia_Row_ID'] = None
                for col in ['Maestro_ID', 'VIN_Year_Min', 'VIN_Year_Max', 'Equivalencia_Row_ID']:
                    if col in entry and pd.notna(entry[col]):
                        try:
                            entry[col] = int(entry[col])
                        except (ValueError, TypeError):
                            entry[col] = None
                    elif col in entry:
                        entry[col] = None
                if 'Confidence' in entry and pd.notna(entry['Confidence']):
                    try:
                        entry['Confidence'] = float(entry['Confidence'])
                    except (ValueError, TypeError):
                        entry['Confidence'] = None
                elif 'Confidence' in entry:
                    entry['Confidence'] = None
                maestro_list.append(entry)
            print(f"Loaded {len(maestro_list)} records from Maestro.xlsx.")
            return maestro_list
        except Exception as e:
            print(
                f"Error loading or processing Maestro.xlsx: {e}. Maestro data will be empty.")
            return []

    # Removed load_vin_details_from_chunks

    def create_widgets(self):
        """Creates the GUI widgets."""
        # --- Top Frame with two columns ---
        top_frame = ttk.Frame(self.root, padding=(10, 5))
        top_frame.pack(padx=10, pady=10, fill="x", expand=False)

        # Configure columns for 60/40 split
        top_frame.columnconfigure(0, weight=60)  # Left column (60%)
        top_frame.columnconfigure(1, weight=40)  # Right column (40%)

        # --- Input Frame (Left Column) ---
        input_frame = ttk.LabelFrame(top_frame, text="Input", padding=(10, 5))
        input_frame.grid(row=0, column=0, padx=(0, 5), pady=5, sticky="nsew")

        # VIN Input
        ttk.Label(input_frame, text="VIN (17 characters):").grid(
            row=0, column=0, padx=5, pady=5, sticky="w")
        self.vin_entry = ttk.Entry(input_frame, width=25)
        self.vin_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        # Part Descriptions Input
        ttk.Label(input_frame, text="Part Descriptions (one per line):").grid(
            row=1, column=0, padx=5, pady=5, sticky="nw")
        self.parts_text = tk.Text(
            input_frame, width=40, height=5)  # Reduced width
        self.parts_text.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        input_frame.columnconfigure(1, weight=1)  # Allow parts_text to expand

        # Find SKUs Button
        self.find_button = ttk.Button(
            input_frame, text="Find SKUs", command=self.find_skus_handler)
        self.find_button.grid(row=2, column=1, padx=5, pady=10, sticky="e")

        # --- Vehicle Details Frame (Right Column) ---
        self.vehicle_details_frame = ttk.LabelFrame(
            top_frame, text="Predicted Vehicle Details", padding=(10, 5))
        self.vehicle_details_frame.grid(
            row=0, column=1, padx=(5, 0), pady=5, sticky="nsew")

        # Placeholder for vehicle details (will be populated after prediction)
        self.vehicle_details_placeholder = ttk.Label(
            self.vehicle_details_frame, text="Vehicle details will appear here after VIN prediction.")
        self.vehicle_details_placeholder.pack(padx=5, pady=5, anchor="nw")

        # --- Output Frame ---
        output_frame = ttk.LabelFrame(
            # Changed title to be more specific
            self.root, text="SKU Suggestions", padding=(10, 5))
        output_frame.pack(padx=10, pady=10, fill="both", expand=True)

        # Make the output frame resize correctly
        self.root.rowconfigure(1, weight=1)
        self.root.columnconfigure(0, weight=1)
        output_frame.rowconfigure(0, weight=1)
        output_frame.columnconfigure(0, weight=1)

        # Scrollable Canvas for results
        canvas = tk.Canvas(output_frame)
        scrollbar = ttk.Scrollbar(
            output_frame, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)

        # Configure the scrollable frame to expand to fill the canvas width
        self.scrollable_frame.columnconfigure(0, weight=1)

        # Bind the frame to update the scrollregion when its size changes
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        # Enable mousewheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # Create window with scrollable frame and configure it to expand horizontally
        canvas.create_window((0, 0), window=self.scrollable_frame,
                             anchor="nw", width=canvas.winfo_width())
        canvas.configure(yscrollcommand=scrollbar.set)

        # Make sure the canvas expands to fill the frame
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        # Update the canvas when it's resized to adjust the scrollable frame width
        def _on_canvas_configure(event):
            canvas.itemconfig(canvas.find_withtag("all")[0], width=event.width)

        canvas.bind("<Configure>", _on_canvas_configure)

        # Placeholder Label inside scrollable frame
        self.results_placeholder_label = ttk.Label(
            self.scrollable_frame, text="Enter VIN and Descriptions, then click 'Find SKUs'.")
        self.results_placeholder_label.pack(padx=5, pady=5, anchor="nw")

        # --- Bottom Frame for Save Button ---
        bottom_frame = ttk.Frame(self.root, padding=(10, 5))
        bottom_frame.pack(side=tk.BOTTOM, fill="x",
                          expand=False, pady=(0, 10), padx=10)

        self.save_button = ttk.Button(
            bottom_frame, text="Save Confirmed Selections", command=self.save_selections_handler, state=tk.DISABLED)
        self.save_button.pack()  # Pack inside the bottom_frame

        # Instance variables
        self.vehicle_details = None
        self.processed_parts = None
        self.current_suggestions = {}
        self.selection_vars = {}
        self.part_frames_widgets = []  # To store part_frame widgets for responsive layout
        self.current_num_columns = 0  # To track current number of columns in results

        # Removed manual input variables

    def _get_sku_nn_prediction(self, make: str, model_year: str, series: str, description: str) -> str | None:
        """
        Uses the loaded SKU NN model to predict an SKU.
        Returns the predicted SKU string or None if prediction fails or model not available.
        """
        if not sku_nn_model or not sku_nn_encoder_make or not sku_nn_encoder_model_year or \
           not sku_nn_encoder_series or not sku_nn_tokenizer_desc or not sku_nn_encoder_sku:
            print(
                "SKU NN model or one of its preprocessors is not loaded. Skipping NN prediction.")
            return None

        try:
            # Use our predict_sku function from sku_nn_pytorch.py
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

            # Create a dictionary of encoders to pass to predict_sku
            encoders = {
                'Make': sku_nn_encoder_make,
                'Model Year': sku_nn_encoder_model_year,
                'Series': sku_nn_encoder_series,
                'tokenizer': sku_nn_tokenizer_desc,
                'sku': sku_nn_encoder_sku
            }

            # Call the predict_sku function
            predicted_sku, confidence = predict_sku(
                model=sku_nn_model,
                encoders=encoders,
                make=make,
                model_year=model_year,
                series=series,
                description=description,
                device=device
            )

            if predicted_sku:
                print(
                    f"  SKU NN Prediction for '{description}': {predicted_sku} (Confidence: {confidence:.4f})")
                return predicted_sku, confidence
            else:
                print(f"  SKU NN Prediction failed for '{description}'")
                return None

        except ValueError as ve:
            # This can happen if a category (Make, Year, Series) was not seen during training
            print(
                f"  SKU NN Prediction Error: Could not encode inputs for '{description}'. Untrained category? Details: {ve}")
            return None
        except Exception as e:
            print(f"  Error during SKU NN prediction for '{description}': {e}")
            return None

    def find_skus_handler(self):
        """
        Handles the 'Find SKUs' button click.
        Uses trained models to predict VIN details, processes parts, searches, and displays.
        """
        print("\n--- 'Find SKUs' button clicked ---")
        vin = self.vin_entry.get().strip().upper()

        # Clear previous results
        self._clear_results_area()

        # Clear vehicle details frame
        for widget in self.vehicle_details_frame.winfo_children():
            widget.destroy()
        ttk.Label(self.vehicle_details_frame,
                  text="Processing VIN...").pack(anchor="w", padx=5, pady=5)

        print(f"VIN Entered: {vin}")

        # Validate VIN format
        if not vin or len(vin) != 17 or not re.match("^[A-HJ-NPR-Z0-9]{17}$", vin):
            messagebox.showerror(
                "Invalid VIN", "VIN must be 17 alphanumeric characters (excluding I, O, Q).")
            ttk.Label(self.scrollable_frame,
                      text="Error: Invalid VIN format.").pack(anchor="nw")
            return

        # --- Predict VIN Details using Models ---
        predicted_details = self.predict_vin_details(vin)

        if not predicted_details:
            ttk.Label(self.scrollable_frame,
                      text=f"Could not predict details for VIN: {vin}.").pack(anchor="nw")
            # Optionally allow manual input here if prediction fails completely
            # self._prompt_for_manual_details(vin) # If we want fallback to manual
            return

        self.vehicle_details = predicted_details  # Store predicted details
        print(f"Predicted Vehicle Details: {self.vehicle_details}")

        # Proceed with part processing and search using predicted details
        self._process_parts_and_continue_search()

    def predict_vin_details(self, vin: str) -> dict | None:
        """Predicts Make, Year, Series using loaded models."""
        if not model_maker or not model_year or not model_series:
            print("Error: Prediction models not loaded.")
            messagebox.showerror(
                "Prediction Error", "VIN prediction models are not loaded. Cannot proceed.")
            return None

        features = extract_vin_features(vin)
        if not features:
            messagebox.showerror("Prediction Error",
                                 "Could not extract features from VIN.")
            return None

        details = {"Make": "N/A", "Model Year": "N/A",
                   "Series": "N/A", "Model": "N/A", "Body Class": "N/A"}

        try:
            # Predict Maker
            wmi_encoded = encoder_x_maker.transform(
                np.array([[features['wmi']]]))
            # Check for unknown category before prediction
            if -1 in wmi_encoded:
                details['Make'] = "Unknown (WMI)"
            else:
                maker_pred_encoded = model_maker.predict(wmi_encoded)
                # Check for unknown category in prediction output (shouldn't happen with CategoricalNB if input known)
                if maker_pred_encoded[0] != -1:
                    details['Make'] = encoder_y_maker.inverse_transform(
                        maker_pred_encoded.reshape(-1, 1))[0]
                else:  # Should not happen if input was known, but handle defensively
                    details['Make'] = "Unknown (Prediction)"

            # Predict Year
            year_code_encoded = encoder_x_year.transform(
                np.array([[features['year_code']]]))
            if -1 in year_code_encoded:
                details['Model Year'] = "Unknown (Code)"
            else:
                year_pred_encoded = model_year.predict(year_code_encoded)
                if year_pred_encoded[0] != -1:
                    details['Model Year'] = encoder_y_year.inverse_transform(
                        year_pred_encoded.reshape(-1, 1))[0]
                else:
                    # Fallback to direct map if model fails (unlikely for year)
                    details['Model Year'] = str(decode_year(features['year_code'])) if decode_year(
                        features['year_code']) else "Unknown (Code)"

            # Predict Series
            series_features_encoded = encoder_x_series.transform(
                np.array([[features['wmi'], features['vds_full']]]))
            # Check if either feature was unknown
            if -1 in series_features_encoded[0]:
                details['Series'] = "Unknown (VDS/WMI)"
            else:
                series_pred_encoded = model_series.predict(
                    series_features_encoded)
                if series_pred_encoded[0] != -1:
                    details['Series'] = encoder_y_series.inverse_transform(
                        series_pred_encoded.reshape(-1, 1))[0]
                else:
                    details['Series'] = "Unknown (Prediction)"

            # Model and Body Class are not predicted by these models
            details['Model'] = "N/A (Not Predicted)"
            details['Body Class'] = "N/A (Not Predicted)"

            return details

        except Exception as e:
            print(f"Error during VIN prediction: {e}")
            # Handle potential errors if a feature wasn't seen during training
            messagebox.showwarning(
                "Prediction Warning", f"Could not reliably predict all details for VIN: {e}")
            # Return partially filled details if possible
            return details

    def _clear_results_area(self):
        """Clears the widgets in the scrollable results frame."""
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        # Removed manual input widget clearing
        self.results_placeholder_label = None
        self.save_button.config(state=tk.DISABLED)

    # Removed _prompt_for_manual_details
    # Removed _handle_manual_details_continue

    def _process_parts_and_continue_search(self):
        """Processes part descriptions and triggers the search and display."""
        # (Content remains largely the same, uses self.vehicle_details which is now predicted)
        part_descriptions_raw = self.parts_text.get("1.0", tk.END).strip()
        self.processed_parts = []
        if part_descriptions_raw:
            original_descriptions = [
                line.strip() for line in part_descriptions_raw.splitlines() if line.strip()]
            print(f"Original Descriptions List: {original_descriptions}")
            for original_desc in original_descriptions:
                normalized_desc = normalize_text(original_desc)
                equivalencia_id = equivalencias_map_global.get(normalized_desc)
                self.processed_parts.append({
                    "original": original_desc,
                    "normalized": normalized_desc,
                    "equivalencia_id": equivalencia_id
                })
                print(
                    f"Processed: '{original_desc}' -> Normalized: '{normalized_desc}', EqID: {equivalencia_id}")
        else:
            print("No part descriptions entered.")
            self.processed_parts = []

        # --- Phase 4: Search Logic ---
        self.current_suggestions = {}
        self.selection_vars = {}

        db_conn = None
        try:
            print(f"Connecting to database: {DEFAULT_DB_PATH}")
            if not os.path.exists(DEFAULT_DB_PATH):
                messagebox.showerror(
                    "Database Error", f"Database file not found at {DEFAULT_DB_PATH}. Please run the offline processor first.")
                ttk.Label(self.scrollable_frame, text=f"Error: Database not found at {DEFAULT_DB_PATH}").pack(
                    anchor="nw")
                return
            db_conn = sqlite3.connect(DEFAULT_DB_PATH)
            cursor = db_conn.cursor()
            print("Database connection successful.")

            for part_info in self.processed_parts:
                original_desc = part_info["original"]
                normalized_desc = part_info["normalized"]
                eq_id = part_info["equivalencia_id"]
                print(
                    f"\nSearching for: '{original_desc}' (Normalized: '{normalized_desc}', EqID: {eq_id})")

                suggestions = {}

                # Use self.vehicle_details which is now PREDICTED
                predicted_make_val = self.vehicle_details.get('Make', 'N/A')
                if isinstance(predicted_make_val, np.ndarray):
                    vin_make = str(predicted_make_val.item()).upper(
                    ) if predicted_make_val.size > 0 else 'N/A'
                else:
                    vin_make = str(predicted_make_val).upper() if pd.notna(
                        predicted_make_val) else 'N/A'

                # Model is likely N/A from predictor
                vin_model = self.vehicle_details.get('Model', 'N/A')
                vin_year_str = self.vehicle_details.get('Model Year', 'N/A')
                vin_year = None  # Initialize
                # Check if it's still an array element
                if isinstance(vin_year_str, np.ndarray):
                    vin_year_str_scalar = vin_year_str.item()  # Extract scalar value
                else:
                    vin_year_str_scalar = vin_year_str  # Use as is if already scalar/string

                if vin_year_str_scalar and vin_year_str_scalar != 'N/A':
                    try:
                        vin_year = int(vin_year_str_scalar)
                    except (ValueError, TypeError):
                        print(
                            f"Warning: Could not convert predicted year '{vin_year_str_scalar}' to integer.")
                        vin_year = None  # Ensure it's None if conversion fails

                # Maestro Search (Adjust matching if needed, e.g., ignore Model if not predicted)
                if eq_id is not None:
                    print(f"  Searching Maestro (EqID: {eq_id})...")
                    for maestro_entry in maestro_data_global:
                        # Match on Make, Year, EqID (Model might be unreliable from predictor)
                        match = (maestro_entry.get('Equivalencia_Row_ID') == eq_id and
                                 maestro_entry.get('VIN_Make') == vin_make and
                                 # Match on year string
                                 str(maestro_entry.get('VIN_Year_Min')) == vin_year_str)
                        if match:
                            sku = maestro_entry.get('Confirmed_SKU')
                            if sku and sku not in suggestions:
                                suggestions[sku] = {
                                    "confidence": 1.0, "source": "Maestro"}
                                print(
                                    f"    Found in Maestro: {sku} (Conf: 1.0)")

                # SQLite Search (ID Match - Adjust matching)
                if eq_id is not None and vin_year is not None:
                    print(
                        f"  Searching SQLite DB (EqID: {eq_id}, Year: {vin_year})...")
                    try:
                        # Query without Model if it's not reliably predicted
                        cursor.execute("""
                            SELECT sku, COUNT(*) as frequency
                            FROM historical_parts
                            WHERE vin_make = ? AND vin_year = ? AND Equivalencia_Row_ID = ?
                            GROUP BY sku
                        """, (vin_make, vin_year, eq_id))  # Removed vin_model from WHERE
                        results = cursor.fetchall()
                        total_matches = sum(row[1] for row in results)
                        for sku, frequency in results:
                            if sku not in suggestions:
                                confidence = round(
                                    0.5 + 0.4 * (frequency / total_matches), 3) if total_matches > 0 else 0.5
                                suggestions[sku] = {
                                    "confidence": confidence, "source": f"DB (ID:{eq_id})"}
                                print(
                                    f"    Found in DB (ID Match): {sku} (Freq: {frequency}, Conf: {confidence})")
                    except Exception as db_err:
                        print(
                            f"    Error querying SQLite DB (ID Match): {db_err}")

                # Fallback Search (Adjust matching)
                if eq_id is None:  # This block is for fallback if no Equivalencia_Row_ID
                    print(
                        f"  Fallback Search (Normalized: '{normalized_desc}')...")
                    # Fallback in Maestro
                    for maestro_entry in maestro_data_global:
                        # Match on Make, Year, Normalized Desc
                        if (maestro_entry.get('Normalized_Description_Input') == normalized_desc and
                            maestro_entry.get('VIN_Make') == vin_make and
                                str(maestro_entry.get('VIN_Year_Min')) == vin_year_str):
                            sku = maestro_entry.get('Confirmed_SKU')
                            if sku and sku not in suggestions:
                                suggestions[sku] = {
                                    "confidence": 0.2, "source": "Maestro (Fallback)"}
                                print(
                                    f"    Found in Maestro (Fallback): {sku} (Conf: 0.2)")
                    # Fallback in SQLite
                    if vin_year is not None:
                        try:
                            # Query without Model
                            cursor.execute("""
                                SELECT sku, COUNT(*) as frequency
                                FROM historical_parts
                                WHERE vin_make = ? AND vin_year = ? AND normalized_description = ?
                                GROUP BY sku
                            """, (vin_make, vin_year, normalized_desc))  # Removed vin_model
                            results = cursor.fetchall()
                            for sku, frequency in results:
                                if sku not in suggestions:
                                    suggestions[sku] = {
                                        "confidence": 0.1, "source": f"DB (Text Fallback)"}
                                    print(
                                        f"    Found in DB (Fallback): {sku} (Conf: 0.1)")
                        except Exception as db_err:
                            print(
                                f"    Error querying SQLite DB (Fallback): {db_err}")

                # --- Add SKU NN Prediction ---
                # Ensure vehicle details are strings for the NN model
                # vin_make is already a string
                # vin_year_str_scalar is already a string

                predicted_series_val = self.vehicle_details.get(
                    'Series', 'N/A')
                if isinstance(predicted_series_val, np.ndarray) and predicted_series_val.size > 0:
                    vin_series_str_for_nn = str(predicted_series_val.item())
                elif pd.notna(predicted_series_val):
                    vin_series_str_for_nn = str(predicted_series_val)
                else:
                    vin_series_str_for_nn = "N/A"

                if vin_make != 'N/A' and vin_year_str_scalar != 'N/A' and vin_series_str_for_nn != 'N/A':
                    print(
                        f"  Attempting SKU NN prediction for: Make='{vin_make}', Year='{vin_year_str_scalar}', Series='{vin_series_str_for_nn}', Desc='{original_desc}'")
                    sku_nn_output = self._get_sku_nn_prediction(
                        make=vin_make,
                        model_year=vin_year_str_scalar,  # Already a string
                        series=vin_series_str_for_nn,
                        description=original_desc  # Helper will normalize
                    )
                    if sku_nn_output:
                        nn_sku, nn_confidence = sku_nn_output
                        if nn_sku and nn_sku not in suggestions:  # Add if new
                            suggestions[nn_sku] = {"confidence": float(
                                nn_confidence), "source": "SKU-NN"}
                            print(
                                f"    Found via SKU-NN: {nn_sku} (Conf: {nn_confidence:.4f})")
                        elif nn_sku and nn_sku in suggestions and suggestions[nn_sku]["source"] != "Maestro":
                            # Optionally update if NN confidence is higher than other non-Maestro sources
                            if float(nn_confidence) > suggestions[nn_sku]["confidence"]:
                                suggestions[nn_sku] = {"confidence": float(
                                    nn_confidence), "source": "SKU-NN (Update)"}
                                print(
                                    f"    Updated via SKU-NN: {nn_sku} (New Conf: {nn_confidence:.4f})")
                else:
                    print(
                        "  Skipping SKU NN prediction due to missing Make, Year, or Series from VIN prediction.")
                # --- End SKU NN Prediction ---

                sorted_suggestions = sorted(
                    suggestions.items(), key=lambda item: item[1]['confidence'], reverse=True)
                self.current_suggestions[original_desc] = sorted_suggestions
                print(
                    f"  Suggestions for '{original_desc}': {sorted_suggestions}")

        except Exception as e:
            messagebox.showerror(
                "Search Error", f"An error occurred during search: {e}")
            print(f"Error during search phase: {e}")
            ttk.Label(self.scrollable_frame,
                      text=f"Error during search: {e}").pack(anchor="nw")
        finally:
            if db_conn:
                db_conn.close()
                print("Database connection closed.")

        # --- Update Results Display ---
        self.display_results()  # Call method to update GUI

    def display_results(self):
        """Updates the vehicle details frame and scrollable results frame with SKU suggestions."""
        # Clear previous results
        self._clear_results_area()

        # Update Vehicle Details in the right column frame
        # First, clear any existing content in the vehicle details frame
        for widget in self.vehicle_details_frame.winfo_children():
            widget.destroy()

        # Display Vehicle Details in the right column frame
        if self.vehicle_details:
            vin = self.vin_entry.get().strip().upper()
            ttk.Label(self.vehicle_details_frame, text=f"VIN: {vin}").pack(
                anchor="w", padx=5, pady=2)
            ttk.Label(
                self.vehicle_details_frame, text=f"Predicted Make: {self.vehicle_details.get('Make', 'N/A')}").pack(anchor="w", padx=5, pady=2)
            ttk.Label(
                self.vehicle_details_frame, text=f"Predicted Year: {self.vehicle_details.get('Model Year', 'N/A')}").pack(anchor="w", padx=5, pady=2)
            ttk.Label(
                self.vehicle_details_frame, text=f"Predicted Series: {self.vehicle_details.get('Series', 'N/A')}").pack(anchor="w", padx=5, pady=2)
        else:
            ttk.Label(self.vehicle_details_frame,
                      text="Vehicle details could not be predicted.").pack(anchor="w", padx=5, pady=5)

        # Display SKU Suggestions (Tasks 5.2, 5.3, 5.4) - Remains the same logic
        if not self.processed_parts:
            ttk.Label(self.scrollable_frame,
                      text="\nNo part descriptions were entered.").pack(anchor="nw")
        elif not self.current_suggestions and self.processed_parts:
            ttk.Label(self.scrollable_frame,
                      text="\nNo suggestions found for the entered descriptions.").pack(anchor="nw")
        elif self.current_suggestions:
            # Add a small separator but no redundant header
            ttk.Separator(self.scrollable_frame, orient='horizontal').pack(
                fill='x', pady=5)

            # Create a container frame for the responsive grid
            self.results_grid_container = ttk.Frame(self.scrollable_frame)
            self.results_grid_container.pack(
                fill="both", expand=True, padx=5, pady=5)  # Fill both to get width

            # Set a minimum width to ensure proper layout calculation
            self.results_grid_container.config(
                width=self.root.winfo_width() - 50)

            # Clear previous part frames widgets list
            self.part_frames_widgets = []

            # Reset the current number of columns to force recalculation
            self.current_num_columns = 0

            # Create frames for each part description
            for part_info in enumerate(self.processed_parts):
                # Use index 1 to get the actual part_info dict
                original_desc = part_info[1]["original"]
                suggestions_list = self.current_suggestions.get(
                    original_desc, [])

                # Create part_frame but do not grid it yet. It will be gridded by _resize_results_columns
                part_frame = ttk.LabelFrame(
                    self.results_grid_container, text=f"{original_desc}", padding=5)
                self.part_frames_widgets.append(part_frame)

                # Configure internal column of part_frame
                part_frame.columnconfigure(0, weight=1)

                # Set a minimum width for consistent sizing
                part_frame.config(width=200)

                if suggestions_list:
                    self.selection_vars[original_desc] = tk.StringVar(
                        value=None)
                    for sku, info in suggestions_list:
                        radio_text = f"SKU: {sku} (Conf: {info['confidence']:.3f}, Src: {info['source']})"
                        rb = ttk.Radiobutton(
                            part_frame,
                            text=radio_text,
                            variable=self.selection_vars[original_desc],
                            value=sku
                        )
                        rb.pack(anchor="w", padx=5, pady=2, fill="x")
                    rb_none = ttk.Radiobutton(
                        part_frame,
                        text="None of these / Manual Entry",
                        variable=self.selection_vars[original_desc],
                        value=""
                    )
                    rb_none.pack(anchor="w", padx=5, pady=2, fill="x")
                else:
                    ttk.Label(part_frame, text="  (No suggestions found)").pack(
                        anchor="w", padx=15, pady=5)

            # Initial layout + bind configure event for responsiveness
            print("Initializing responsive layout...")
            self.root.update_idletasks()  # Force geometry update before calculating layout
            self._resize_results_columns()  # Perform initial layout

            # Bind to both the container and root window resize events
            self.results_grid_container.bind(
                "<Configure>", self._on_results_configure)
            self.root.bind("<Configure>", self._on_results_configure)

            if any(self.current_suggestions.values()):
                self.save_button.config(state=tk.NORMAL)
            else:
                self.save_button.config(state=tk.DISABLED)

    def _on_results_configure(self, event=None):
        # This method is called when the results_grid_container is resized
        # We add a small delay (debounce) to avoid excessive re-layouts during rapid resizing
        if hasattr(self, '_after_id_resize'):
            self.root.after_cancel(self._after_id_resize)
        self._after_id_resize = self.root.after(
            100, self._resize_results_columns)

    def _resize_results_columns(self):
        """Recalculates and applies the grid layout for part_frames to create a responsive layout."""
        if not hasattr(self, 'results_grid_container') or not self.results_grid_container.winfo_exists():
            return
        if not self.part_frames_widgets:  # No items to grid
            return

        # Get the current width of the container
        container_width = self.results_grid_container.winfo_width()

        # If the container width is not yet initialized, use the root window width as a fallback
        if container_width <= 1:
            container_width = self.root.winfo_width() - 40  # Subtract some padding

        # Get the actual width of a part frame by measuring the first one
        # This ensures we use the real width rather than an estimate
        if self.part_frames_widgets:
            # Update the first widget to ensure its size is calculated
            self.part_frames_widgets[0].update_idletasks()
            # Add padding
            actual_item_width = self.part_frames_widgets[0].winfo_reqwidth(
            ) + 10
        else:
            actual_item_width = 220  # Fallback if no widgets exist

        # Calculate the number of columns that can fit completely
        # We only want to show complete columns (no partial columns)
        if container_width <= actual_item_width:
            num_columns = 1
        else:
            # Calculate how many complete columns can fit
            num_columns = max(1, int(container_width / actual_item_width))

            # Ensure we don't create more columns than we have items
            num_columns = min(num_columns, len(self.part_frames_widgets))

        print(
            f"Container width: {container_width}, Item width: {actual_item_width}, Complete columns: {num_columns}")

        # Check if we need to update the layout
        layout_needs_update = (
            num_columns != self.current_num_columns or
            self.current_num_columns == 0 or
            len(self.results_grid_container.grid_slaves()) != len(
                self.part_frames_widgets)
        )

        if not layout_needs_update:
            # Further check if all columns have the correct weight
            all_weights_correct = True
            for i in range(num_columns):
                if self.results_grid_container.grid_columnconfigure(i).get('weight', '0') != '1':
                    all_weights_correct = False
                    break

            if all_weights_correct:
                return  # No layout change needed

        # Update the layout

        # 1. Clear all existing column configurations
        current_configured_cols = max(self.current_num_columns,
                                      self.results_grid_container.grid_size()[0])
        for i in range(current_configured_cols):
            self.results_grid_container.columnconfigure(i, weight=0)

        # 2. Remove all widgets from the grid
        for widget in self.results_grid_container.grid_slaves():
            widget.grid_forget()

        # 3. Configure the new columns with equal weight
        for i in range(num_columns):
            self.results_grid_container.columnconfigure(i, weight=1)

        # 4. Update the current number of columns
        self.current_num_columns = num_columns

        # 5. Re-grid all the part frames in the new layout
        for idx, frame_widget in enumerate(self.part_frames_widgets):
            row = idx // num_columns
            col = idx % num_columns
            frame_widget.grid(row=row, column=col, padx=5,
                              pady=5, sticky="nsew")

        print(
            f"Updated layout to {num_columns} complete columns with {len(self.part_frames_widgets)} items")

    def save_selections_handler(self):
        """
        Handles the 'Save Confirmed Selections' button click.
        Gathers selected SKUs, adds them to the in-memory Maestro data,
        and writes the updated data back to Maestro.xlsx.
        (Corresponds to Tasks 5.6, 5.7, 5.8)
        """
        global maestro_data_global
        print("\n--- 'Save Confirmed Selections' clicked ---")
        if not self.vehicle_details or not self.processed_parts:
            messagebox.showwarning(
                "Cannot Save", "No valid search results available to save.")
            return

        selections_to_save = []
        for part_info in self.processed_parts:
            original_desc = part_info["original"]
            selected_sku_var = self.selection_vars.get(original_desc)
            if selected_sku_var:
                selected_sku = selected_sku_var.get()
                if selected_sku:
                    print(
                        f"Selected for '{original_desc}': SKU = {selected_sku}")
                    part_data = next(
                        (p for p in self.processed_parts if p["original"] == original_desc), None)
                    if part_data:
                        selections_to_save.append({
                            "vin_details": self.vehicle_details,
                            "original_description": original_desc,
                            "normalized_description": part_data["normalized"],
                            "equivalencia_id": part_data["equivalencia_id"],
                            "confirmed_sku": selected_sku
                        })
                else:
                    print(f"Selected for '{original_desc}': None")
            else:
                print(f"No selection variable found for '{original_desc}'")

        if not selections_to_save:
            messagebox.showinfo(
                "Nothing to Save", "No specific SKUs were selected for confirmation.")
            return

        added_count = 0
        skipped_count = 0
        max_id = 0
        if maestro_data_global:
            ids = [entry.get('Maestro_ID', 0) for entry in maestro_data_global if isinstance(
                entry.get('Maestro_ID'), int)]
            if ids:
                max_id = max(ids)
        next_id = max_id + 1

        for selection in selections_to_save:
            is_duplicate = False
            for existing_entry in maestro_data_global:
                # Check for duplicates based on predicted details and selection
                if (existing_entry.get('VIN_Make') == selection['vin_details'].get('Make') and
                    existing_entry.get('VIN_Model') == selection['vin_details'].get('Model') and
                    existing_entry.get('VIN_Year_Min') == selection['vin_details'].get('Model Year') and
                    existing_entry.get('Normalized_Description_Input') == selection['normalized_description'] and
                        existing_entry.get('Confirmed_SKU') == selection['confirmed_sku']):
                    is_duplicate = True
                    break
            if not is_duplicate:
                new_entry = {
                    'Maestro_ID': next_id,
                    'VIN_Make': selection['vin_details'].get('Make'),
                    'VIN_Model': selection['vin_details'].get('Model'),
                    'VIN_Year_Min': selection['vin_details'].get('Model Year'),
                    'VIN_Year_Max': selection['vin_details'].get('Model Year'),
                    'VIN_Series_Trim': selection['vin_details'].get('Series'),
                    'VIN_BodyStyle': selection['vin_details'].get('Body Class'),
                    'Original_Description_Input': selection['original_description'],
                    'Normalized_Description_Input': selection['normalized_description'],
                    'Equivalencia_Row_ID': selection['equivalencia_id'],
                    'Confirmed_SKU': selection['confirmed_sku'],
                    'Confidence': 1.0,
                    'Source': 'UserConfirmed',
                    'Date_Added': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                maestro_data_global.append(new_entry)
                added_count += 1
                next_id += 1
            else:
                skipped_count += 1
                print(
                    f"Skipped duplicate: {selection['original_description']} - {selection['confirmed_sku']}")

        if added_count > 0:
            print(
                f"Attempting to save {added_count} new entries to {DEFAULT_MAESTRO_PATH}...")
            try:
                maestro_columns = [
                    'Maestro_ID', 'VIN_Make', 'VIN_Model', 'VIN_Year_Min', 'VIN_Year_Max',
                    'VIN_Series_Trim', 'VIN_BodyStyle', 'Original_Description_Input',
                    'Normalized_Description_Input', 'Equivalencia_Row_ID', 'Confirmed_SKU',
                    'Confidence', 'Source', 'Date_Added'
                ]
                df_to_save = pd.DataFrame(
                    maestro_data_global, columns=maestro_columns)
                df_to_save.to_excel(DEFAULT_MAESTRO_PATH, index=False)
                messagebox.showinfo(
                    "Save Successful", f"{added_count} new confirmation(s) saved to Maestro.xlsx.")
                print(
                    f"Successfully saved {added_count} new entries. Total Maestro entries: {len(maestro_data_global)}")
            except Exception as e:
                messagebox.showerror(
                    "Save Error", f"Failed to write to Maestro.xlsx: {e}")
                print(f"Error writing Maestro.xlsx: {e}")
        elif skipped_count > 0:
            messagebox.showinfo(
                "Already Saved", "The selected confirmations were already present in Maestro.xlsx.")
            print(f"Skipped {skipped_count} duplicate entries.")
        else:
            messagebox.showinfo(
                "Nothing Saved", "No new confirmations were added.")

    # Removed decode_vin_nhtsa


if __name__ == '__main__':
    current_dir = os.getcwd()
    print(f"Current working directory: {current_dir}")
    print(
        f"Expected Equivalencias.xlsx at: {os.path.join(current_dir, DEFAULT_EQUIVALENCIAS_PATH)}")
    print(
        f"Expected/Creating Maestro.xlsx at: {os.path.join(current_dir, DEFAULT_MAESTRO_PATH)}")
    print(
        f"Expected fixacar_history.db at: {os.path.join(current_dir, DEFAULT_DB_PATH)}")
    # Removed the print statement that caused the error

    root = tk.Tk()
    root.title("Fixacar SKU Finder v2.0 (with VIN Predictor)")
    root.geometry("1200x800")  # Set a reasonable default size
    app = FixacarApp(root)
    root.mainloop()
