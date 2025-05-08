import tkinter as tk
from tkinter import ttk
from tkinter import messagebox  # For showing error popups
import os
import pandas as pd
import requests  # For NHTSA API
import sqlite3  # For database connection
from collections import defaultdict  # For counting frequencies
# Assuming text_utils is in the same directory or src is in PYTHONPATH
try:
    from utils.text_utils import normalize_text
except ImportError:
    # Fallback for direct execution if src is not in path (e.g. during development)
    from .utils.text_utils import normalize_text


# --- Configuration ---
DEFAULT_EQUIVALENCIAS_PATH = "Equivalencias.xlsx"
# Will be created in Source_Files/data/
DEFAULT_MAESTRO_PATH = "data/Maestro.xlsx"
DEFAULT_DB_PATH = "data/fixacar_history.db"

# In-memory data stores
equivalencias_map_global = {}
maestro_data_global = []


class FixacarApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fixacar SKU Finder v1.0")
        self.root.geometry("800x600")

        # Load initial data
        self.load_all_data()

        # Setup UI
        self.create_widgets()

    def load_all_data(self):
        """Loads all necessary data on startup."""
        global equivalencias_map_global, maestro_data_global

        print("--- Loading Application Data ---")
        equivalencias_map_global = self.load_equivalencias_data(
            DEFAULT_EQUIVALENCIAS_PATH)
        maestro_data_global = self.load_maestro_data(
            DEFAULT_MAESTRO_PATH, equivalencias_map_global)
        # DB connection will be established when needed for search
        print("--- Application Data Loading Complete ---")

    def load_equivalencias_data(self, file_path: str) -> dict:
        """
        Loads Equivalencias.xlsx, normalizes terms, and creates a mapping
        from normalized terms to a generated Equivalencia_Row_ID (1-based row index).
        (Corresponds to TODO Task 2.4)
        """
        print(f"Loading equivalencias from: {file_path}")
        if not os.path.exists(file_path):
            print(
                f"Warning: Equivalencias file not found at {file_path}. Equivalency linking will be disabled.")
            return {}

        try:
            df = pd.read_excel(file_path, sheet_name=0)
            equivalencias_map = {}
            for index, row in df.iterrows():
                equivalencia_row_id = index + 1  # 1-based ID
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
        """
        Loads Maestro.xlsx into an in-memory list of dictionaries.
        Creates the file with headers if it doesn't exist.
        Normalizes descriptions and links Equivalencia_Row_ID.
        (Corresponds to TODO Task 2.5)
        """
        print(f"Loading maestro data from: {file_path}")
        maestro_columns = [
            'Maestro_ID', 'VIN_Make', 'VIN_Model', 'VIN_Year_Min', 'VIN_Year_Max',
            'VIN_Series_Trim', 'VIN_BodyStyle', 'Original_Description_Input',
            'Normalized_Description_Input', 'Equivalencia_Row_ID', 'Confirmed_SKU',
            'Confidence', 'Source', 'Date_Added'
        ]

        # Ensure data directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        if not os.path.exists(file_path):
            print(
                f"Maestro file not found at {file_path}. Creating a new one with headers.")
            df_maestro = pd.DataFrame(columns=maestro_columns)
            try:
                df_maestro.to_excel(file_path, index=False)
                return []  # Return empty list for a new file
            except Exception as e:
                print(
                    f"Error creating Maestro.xlsx: {e}. Maestro data will be empty.")
                return []

        try:
            df_maestro = pd.read_excel(file_path, sheet_name=0)
            if df_maestro.empty:
                print("Maestro file is empty.")
                return []

            maestro_list = []
            for _, row in df_maestro.iterrows():
                entry = row.to_dict()
                # Normalize Original_Description_Input if it exists and store it
                original_desc = entry.get('Original_Description_Input', "")
                if pd.notna(original_desc):
                    normalized_desc = normalize_text(str(original_desc))
                    entry['Normalized_Description_Input'] = normalized_desc
                    # Link Equivalencia_Row_ID
                    entry['Equivalencia_Row_ID'] = equivalencias_map.get(
                        normalized_desc)
                else:
                    entry['Normalized_Description_Input'] = ""
                    entry['Equivalencia_Row_ID'] = None

                maestro_list.append(entry)

            print(f"Loaded {len(maestro_list)} records from Maestro.xlsx.")
            return maestro_list
        except Exception as e:
            print(
                f"Error loading or processing Maestro.xlsx: {e}. Maestro data will be empty.")
            return []

    def create_widgets(self):
        """Creates the GUI widgets."""
        # --- Input Frame ---
        input_frame = ttk.LabelFrame(self.root, text="Input", padding=(10, 5))
        input_frame.pack(padx=10, pady=10, fill="x", expand=False)

        # VIN Input
        ttk.Label(input_frame, text="VIN (17 characters):").grid(
            row=0, column=0, padx=5, pady=5, sticky="w")
        self.vin_entry = ttk.Entry(input_frame, width=40)
        self.vin_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        # Part Descriptions Input
        ttk.Label(input_frame, text="Part Descriptions (one per line):").grid(
            row=1, column=0, padx=5, pady=5, sticky="nw")
        self.parts_text = tk.Text(input_frame, width=60, height=10)
        self.parts_text.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        input_frame.columnconfigure(1, weight=1)  # Allow parts_text to expand

        # Find SKUs Button
        self.find_button = ttk.Button(
            input_frame, text="Find SKUs", command=self.find_skus_handler)
        self.find_button.grid(row=2, column=1, padx=5, pady=10, sticky="e")

        # --- Output Frame (Placeholder for now) ---
        output_frame = ttk.LabelFrame(
            self.root, text="Results", padding=(10, 5))
        output_frame.pack(padx=10, pady=10, fill="both", expand=True)

        self.results_label = ttk.Label(
            output_frame, text="Vehicle details and SKU suggestions will appear here.")
        self.results_label.pack(padx=5, pady=5, anchor="nw")

        # More UI elements will be added in Phase 5 (Output area, selections)

    def find_skus_handler(self):
        """
        Handles the 'Find SKUs' button click.
        This will orchestrate Tasks 3.3 through 3.10.
        """
        print("\n--- 'Find SKUs' button clicked ---")
        vin = self.vin_entry.get().strip().upper()
        part_descriptions_raw = self.parts_text.get("1.0", tk.END).strip()

        print(f"VIN Entered: {vin}")
        print(f"Part Descriptions Raw:\n{part_descriptions_raw}")

        # Task 3.4: Basic VIN format validation
        if not vin or len(vin) != 17:
            messagebox.showerror(
                "Invalid VIN", "VIN must be 17 characters long.")
            self.results_label.config(
                text="Error: VIN must be 17 characters long.")
            return

        self.results_label.config(text=f"Decoding VIN: {vin}...")
        self.root.update_idletasks()  # Update GUI to show "Decoding VIN..."

        # Task 3.5 & 3.6: Call NHTSA VIN Decoder API and Handle Response
        vehicle_details = self.decode_vin_nhtsa(vin)

        if not vehicle_details:
            # decode_vin_nhtsa will show its own error message
            self.results_label.config(
                text=f"Failed to decode VIN: {vin}. Check console for details.")
            return

        # Task 3.7: Store extracted VIN details (vehicle_details is already a dict)
        print(f"Decoded Vehicle Details: {vehicle_details}")

        # Tasks 3.8, 3.9, 3.10: Process part descriptions
        processed_parts = []
        if part_descriptions_raw:
            original_descriptions = [
                line.strip() for line in part_descriptions_raw.splitlines() if line.strip()]
            print(f"Original Descriptions List: {original_descriptions}")
            for original_desc in original_descriptions:
                normalized_desc = normalize_text(original_desc)
                equivalencia_id = equivalencias_map_global.get(
                    normalized_desc)  # Task 3.10
                processed_parts.append({
                    "original": original_desc,
                    "normalized": normalized_desc,
                    "equivalencia_id": equivalencia_id
                })
                print(
                    f"Processed: '{original_desc}' -> Normalized: '{normalized_desc}', EqID: {equivalencia_id}")
        else:
            print("No part descriptions entered.")
            processed_parts = []  # Ensure it's an empty list if no input

        # --- Phase 4: Search Logic ---
        # Dictionary to store suggestions for each original part description
        all_part_suggestions = {}

        # Task 4.1: Establish SQLite connection
        db_conn = None
        try:
            print(f"Connecting to database: {DEFAULT_DB_PATH}")
            if not os.path.exists(DEFAULT_DB_PATH):
                messagebox.showerror(
                    "Database Error", f"Database file not found at {DEFAULT_DB_PATH}. Please run the offline processor first.")
                self.results_label.config(
                    text=f"Error: Database not found at {DEFAULT_DB_PATH}")
                return
            db_conn = sqlite3.connect(DEFAULT_DB_PATH)
            cursor = db_conn.cursor()
            print("Database connection successful.")

            # Task 4.2: Loop through each processed part
            for part_info in processed_parts:
                original_desc = part_info["original"]
                normalized_desc = part_info["normalized"]
                eq_id = part_info["equivalencia_id"]
                print(
                    f"\nSearching for: '{original_desc}' (Normalized: '{normalized_desc}', EqID: {eq_id})")

                suggestions = {}  # sku -> {confidence, source}

                # --- Search Strategy ---
                # Get VIN details for matching (use N/A if missing)
                vin_make = vehicle_details.get('Make', 'N/A')
                vin_model = vehicle_details.get('Model', 'N/A')
                vin_year_str = vehicle_details.get('Model Year', 'N/A')
                try:
                    vin_year = int(
                        vin_year_str) if vin_year_str != 'N/A' else None
                except (ValueError, TypeError):
                    vin_year = None

                # Task 4.3: Maestro Search (Highest Confidence)
                if eq_id is not None:
                    print(f"  Searching Maestro (EqID: {eq_id})...")
                    for maestro_entry in maestro_data_global:
                        # Basic matching: Make, Model, Year range, EqID
                        match = (maestro_entry.get('Equivalencia_Row_ID') == eq_id and
                                 maestro_entry.get('VIN_Make') == vin_make and
                                 maestro_entry.get('VIN_Model') == vin_model)
                        # Add year range check if possible (requires Maestro schema consistency)
                        # For now, skipping complex year range matching in Maestro for simplicity
                        if match:
                            sku = maestro_entry.get('Confirmed_SKU')
                            if sku and sku not in suggestions:
                                suggestions[sku] = {
                                    "confidence": 1.0, "source": "Maestro"}
                                print(
                                    f"    Found in Maestro: {sku} (Conf: 1.0)")

                # Task 4.4 & 4.5: SQLite Search (ID Match)
                if eq_id is not None and vin_year is not None:
                    print(
                        f"  Searching SQLite DB (EqID: {eq_id}, Year: {vin_year})...")
                    try:
                        cursor.execute("""
                            SELECT sku, COUNT(*) as frequency 
                            FROM historical_parts 
                            WHERE vin_make = ? AND vin_model = ? AND vin_year = ? AND Equivalencia_Row_ID = ?
                            GROUP BY sku
                        """, (vin_make, vin_model, vin_year, eq_id))

                        results = cursor.fetchall()
                        total_matches = sum(row[1] for row in results)

                        for sku, frequency in results:
                            if sku not in suggestions:  # Prioritize Maestro confidence
                                # Simple confidence based on frequency
                                confidence = round(
                                    0.5 + 0.4 * (frequency / total_matches), 3) if total_matches > 0 else 0.5
                                suggestions[sku] = {
                                    "confidence": confidence, "source": f"DB (ID:{eq_id})"}
                                print(
                                    f"    Found in DB (ID Match): {sku} (Freq: {frequency}, Conf: {confidence})")
                    except Exception as db_err:
                        print(
                            f"    Error querying SQLite DB (ID Match): {db_err}")

                # Task 4.6: Fallback Search (Normalized Text Match)
                if eq_id is None:
                    print(
                        f"  Fallback Search (Normalized: '{normalized_desc}')...")
                    # Fallback in Maestro
                    for maestro_entry in maestro_data_global:
                        if (maestro_entry.get('Normalized_Description_Input') == normalized_desc and
                            maestro_entry.get('VIN_Make') == vin_make and
                                maestro_entry.get('VIN_Model') == vin_model):
                            sku = maestro_entry.get('Confirmed_SKU')
                            if sku and sku not in suggestions:
                                suggestions[sku] = {
                                    "confidence": 0.2, "source": "Maestro (Fallback)"}
                                print(
                                    f"    Found in Maestro (Fallback): {sku} (Conf: 0.2)")

                    # Fallback in SQLite
                    if vin_year is not None:
                        try:
                            cursor.execute("""
                                SELECT sku, COUNT(*) as frequency 
                                FROM historical_parts 
                                WHERE vin_make = ? AND vin_model = ? AND vin_year = ? AND normalized_description = ?
                                GROUP BY sku
                            """, (vin_make, vin_model, vin_year, normalized_desc))

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

                # Task 4.7 & 4.8: Combine, Rank (already combined, just need to sort)
                sorted_suggestions = sorted(
                    suggestions.items(), key=lambda item: item[1]['confidence'], reverse=True)
                all_part_suggestions[original_desc] = sorted_suggestions
                print(
                    f"  Suggestions for '{original_desc}': {sorted_suggestions}")

        except Exception as e:
            messagebox.showerror(
                "Search Error", f"An error occurred during search: {e}")
            print(f"Error during search phase: {e}")
            self.results_label.config(text=f"Error during search: {e}")
        finally:
            if db_conn:
                db_conn.close()
                print("Database connection closed.")

        # --- Update Results Display ---
        # (This part overlaps with Phase 5 but needed for basic output)
        details_str = f"VIN: {vin}\n"
        details_str += f"Make: {vehicle_details.get('Make', 'N/A')}\n"
        details_str += f"Model: {vehicle_details.get('Model', 'N/A')}\n"
        details_str += f"Year: {vehicle_details.get('Model Year', 'N/A')}\n"
        details_str += f"Series: {vehicle_details.get('Series', 'N/A')}\n"
        details_str += f"Body Style: {vehicle_details.get('Body Class', 'N/A')}\n"

        details_str += "\n--- SKU Suggestions ---\n"
        if not processed_parts:
            details_str += "(No descriptions entered)\n"
        elif not all_part_suggestions:
            details_str += "(No suggestions found for entered descriptions)\n"
        else:
            for original_desc, suggestions_list in all_part_suggestions.items():
                details_str += f"\nFor '{original_desc}':\n"
                if suggestions_list:
                    for sku, info in suggestions_list:
                        details_str += f"  - SKU: {sku} (Confidence: {info['confidence']:.3f}, Source: {info['source']})\n"
                else:
                    details_str += "  (No suggestions found)\n"

        # Update the results label - consider using a Text widget or Treeview for better formatting later
        self.results_label.config(text=details_str)

    def decode_vin_nhtsa(self, vin_number: str) -> dict | None:
        """
        Decodes a VIN using the NHTSA vPIC API.
        Returns a dictionary with selected vehicle details or None on error.
        (Corresponds to Tasks 3.5, 3.6)
        """
        api_url = f"https://vpic.nhtsa.dot.gov/api/vehicles/decodevin/{vin_number}?format=json"
        print(f"Calling NHTSA API: {api_url}")
        try:
            response = requests.get(api_url, timeout=10)  # 10 second timeout
            response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)

            data = response.json()

            if not data or not data.get("Results"):
                messagebox.showerror("VIN Decode Error",
                                     "No results from NHTSA API for this VIN.")
                print("NHTSA API: No results found in response.")
                return None

            results = data["Results"]
            details = {}

            # Extract specific fields as per PRD 3.2 (Make, Model, Year, Series, Body Style)
            # The API response field names might vary. Common ones are used here.
            for item in results:
                variable_name = item.get("Variable", "")
                value = item.get("Value", "")
                if not value or value == "Not Applicable":  # Skip empty or "Not Applicable" values
                    continue

                if variable_name == "Make":
                    details['Make'] = value
                elif variable_name == "Model":
                    details['Model'] = value
                elif variable_name == "Model Year":
                    details['Model Year'] = value
                elif variable_name == "Series":
                    details['Series'] = value
                elif variable_name == "Body Class":
                    # Common field for body style
                    details['Body Class'] = value
                # Add more fields if needed, e.g., Trim
                elif variable_name == "Trim":
                    details['Trim'] = value

            if not details.get('Make') or not details.get('Model') or not details.get('Model Year'):
                messagebox.showwarning(
                    "VIN Decode Incomplete", "NHTSA API did not return Make, Model, or Year for this VIN.")
                print(
                    f"NHTSA API: Incomplete essential details. Full results: {results}")
                # Return partial details if some are found, or None if critical ones are missing
                # For now, let's return what we have if anything was found.

            return details if details else None

        except requests.exceptions.HTTPError as e:
            messagebox.showerror(
                "API Error", f"HTTP error calling NHTSA API: {e.response.status_code}\n{e.response.text[:200]}")
            print(f"NHTSA API HTTPError: {e}")
            return None
        except requests.exceptions.RequestException as e:
            messagebox.showerror(
                "Network Error", f"Error calling NHTSA API: {e}")
            print(f"NHTSA API RequestException: {e}")
            return None
        except json.JSONDecodeError:
            messagebox.showerror("API Response Error",
                                 "Invalid JSON response from NHTSA API.")
            print("NHTSA API: JSONDecodeError parsing response.")
            return None
        except Exception as e:
            messagebox.showerror(
                "VIN Decode Error", f"An unexpected error occurred during VIN decoding: {e}")
            print(f"NHTSA API unexpected error: {e}")
            return None


if __name__ == '__main__':
    # This allows running the script directly from Source_Files/
    # For utils.text_utils to be found, ensure src is in PYTHONPATH or run as module if needed
    # e.g. python -m src.main_app from Source_Files parent.
    # For simplicity now, direct run from Source_Files is assumed with try-except for import.

    # If running from Source_Files directory:
    # Make sure Equivalencias.xlsx is in Source_Files
    # Maestro.xlsx will be created in Source_Files/data/
    # fixacar_history.db should be in Source_Files/data/ (created by offline_processor)

    current_dir = os.getcwd()
    print(f"Current working directory: {current_dir}")
    print(
        f"Expected Equivalencias.xlsx at: {os.path.join(current_dir, DEFAULT_EQUIVALENCIAS_PATH)}")
    print(
        f"Expected/Creating Maestro.xlsx at: {os.path.join(current_dir, DEFAULT_MAESTRO_PATH)}")
    print(
        f"Expected fixacar_history.db at: {os.path.join(current_dir, DEFAULT_DB_PATH)}")

    root = tk.Tk()
    app = FixacarApp(root)
    root.mainloop()
