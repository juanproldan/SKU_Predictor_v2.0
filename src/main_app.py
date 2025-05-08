import tkinter as tk
from tkinter import ttk
from tkinter import messagebox  # For showing error popups
import os
import pandas as pd
import requests  # For NHTSA API
import sqlite3  # For database connection
from collections import defaultdict  # For counting frequencies
import datetime  # For timestamping Maestro entries
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
maestro_data_global = []  # This will hold the list of dictionaries


class FixacarApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fixacar SKU Finder v1.0")
        self.root.geometry("800x600")  # Adjusted default size

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
        # Load maestro data into the global variable
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
        data_dir = os.path.dirname(file_path)
        if data_dir and not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)

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
            # Ensure Maestro_ID is treated as integer where possible, handle NaN/None
            if 'Maestro_ID' in df_maestro.columns:
                df_maestro['Maestro_ID'] = pd.to_numeric(
                    df_maestro['Maestro_ID'], errors='coerce').fillna(0).astype(int)

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

                # Ensure numeric types are correct, handle potential read errors
                for col in ['Maestro_ID', 'VIN_Year_Min', 'VIN_Year_Max', 'Equivalencia_Row_ID']:
                    if col in entry and pd.notna(entry[col]):
                        try:
                            entry[col] = int(entry[col])
                        except (ValueError, TypeError):
                            # Or some other default like 0 or -1
                            entry[col] = None
                    elif col in entry:
                        entry[col] = None  # Handle NaN/None explicitly

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

        # --- Output Frame ---
        output_frame = ttk.LabelFrame(
            self.root, text="Results", padding=(10, 5))
        output_frame.pack(padx=10, pady=10, fill="both", expand=True)

        # Make the output frame resize correctly
        self.root.rowconfigure(1, weight=1)  # Allow output_frame row to expand
        # Allow output_frame column to expand
        self.root.columnconfigure(0, weight=1)
        output_frame.rowconfigure(0, weight=1)
        output_frame.columnconfigure(0, weight=1)

        # Scrollable Canvas for results
        canvas = tk.Canvas(output_frame)
        scrollbar = ttk.Scrollbar(
            output_frame, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)  # Frame inside canvas

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        # Placeholder Label inside scrollable frame (will be removed when results are shown)
        self.results_placeholder_label = ttk.Label(
            self.scrollable_frame, text="Enter VIN and Descriptions, then click 'Find SKUs'.")
        self.results_placeholder_label.pack(padx=5, pady=5, anchor="nw")

        # --- Save Button ---
        self.save_button = ttk.Button(
            self.root, text="Save Confirmed Selections", command=self.save_selections_handler, state=tk.DISABLED)
        self.save_button.pack(pady=10)

        # Instance variables to store results context for saving
        self.vehicle_details = None
        self.processed_parts = None
        # Stores {original_desc: [(sku, info), ...]}
        self.current_suggestions = {}
        self.selection_vars = {}  # Stores {original_desc: tk.StringVar()}

    def find_skus_handler(self):
        """
        Handles the 'Find SKUs' button click.
        Orchestrates VIN decoding, part processing, searching, and displaying results.
        """
        print("\n--- 'Find SKUs' button clicked ---")
        vin = self.vin_entry.get().strip().upper()
        part_descriptions_raw = self.parts_text.get("1.0", tk.END).strip()

        # Clear previous results display
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        self.results_placeholder_label = None  # Remove reference
        self.save_button.config(state=tk.DISABLED)  # Disable save button

        print(f"VIN Entered: {vin}")
        print(f"Part Descriptions Raw:\n{part_descriptions_raw}")

        # Task 3.4: Basic VIN format validation
        if not vin or len(vin) != 17:
            messagebox.showerror(
                "Invalid VIN", "VIN must be 17 characters long.")
            ttk.Label(self.scrollable_frame,
                      text="Error: VIN must be 17 characters long.").pack(anchor="nw")
            return

        # Display status before potentially long operation
        status_label = ttk.Label(
            self.scrollable_frame, text=f"Decoding VIN: {vin}...")
        status_label.pack(anchor="nw")
        self.root.update_idletasks()

        # Task 3.5 & 3.6: Call NHTSA VIN Decoder API and Handle Response
        vehicle_details = self.decode_vin_nhtsa(vin)

        # Clear status label
        status_label.destroy()

        if not vehicle_details:
            # decode_vin_nhtsa will show its own error message
            ttk.Label(self.scrollable_frame,
                      text=f"Failed to decode VIN: {vin}. Check console for details.").pack(anchor="nw")
            return

        # Task 3.7: Store extracted VIN details
        self.vehicle_details = vehicle_details
        print(f"Decoded Vehicle Details: {self.vehicle_details}")

        # Tasks 3.8, 3.9, 3.10: Process part descriptions
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

                vin_make = self.vehicle_details.get('Make', 'N/A')
                vin_model = self.vehicle_details.get('Model', 'N/A')
                vin_year_str = self.vehicle_details.get('Model Year', 'N/A')
                try:
                    vin_year = int(
                        vin_year_str) if vin_year_str != 'N/A' else None
                except (ValueError, TypeError):
                    vin_year = None

                # Maestro Search
                if eq_id is not None:
                    print(f"  Searching Maestro (EqID: {eq_id})...")
                    for maestro_entry in maestro_data_global:
                        match = (maestro_entry.get('Equivalencia_Row_ID') == eq_id and
                                 maestro_entry.get('VIN_Make') == vin_make and
                                 maestro_entry.get('VIN_Model') == vin_model)
                        if match:
                            sku = maestro_entry.get('Confirmed_SKU')
                            if sku and sku not in suggestions:
                                suggestions[sku] = {
                                    "confidence": 1.0, "source": "Maestro"}
                                print(
                                    f"    Found in Maestro: {sku} (Conf: 1.0)")

                # SQLite Search (ID Match)
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

                # Fallback Search
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
        """Updates the scrollable results frame with vehicle details and interactive SKU suggestions."""
        # Clear previous results
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        self.results_placeholder_label = None  # Clear reference

        # Display Vehicle Details (Task 5.1)
        if self.vehicle_details:
            vin = self.vin_entry.get().strip().upper()  # Get VIN again for display
            details_frame = ttk.LabelFrame(
                self.scrollable_frame, text="Vehicle Details", padding=5)
            details_frame.pack(pady=5, padx=5, fill="x", anchor="nw")
            ttk.Label(details_frame, text=f"VIN: {vin}").pack(anchor="w")
            ttk.Label(
                details_frame, text=f"Make: {self.vehicle_details.get('Make', 'N/A')}").pack(anchor="w")
            ttk.Label(
                details_frame, text=f"Model: {self.vehicle_details.get('Model', 'N/A')}").pack(anchor="w")
            ttk.Label(
                details_frame, text=f"Year: {self.vehicle_details.get('Model Year', 'N/A')}").pack(anchor="w")
            ttk.Label(
                details_frame, text=f"Series: {self.vehicle_details.get('Series', 'N/A')}").pack(anchor="w")
            ttk.Label(
                details_frame, text=f"Body Style: {self.vehicle_details.get('Body Class', 'N/A')}").pack(anchor="w")
        else:
            ttk.Label(self.scrollable_frame,
                      text="Vehicle details could not be retrieved.").pack(anchor="nw")

        # Display SKU Suggestions (Tasks 5.2, 5.3, 5.4)
        if not self.processed_parts:
            ttk.Label(self.scrollable_frame,
                      text="\nNo part descriptions were entered.").pack(anchor="nw")
        elif not self.current_suggestions:
            ttk.Label(self.scrollable_frame,
                      text="\nNo suggestions found for the entered descriptions.").pack(anchor="nw")
        else:
            ttk.Separator(self.scrollable_frame, orient='horizontal').pack(
                fill='x', pady=10)
            ttk.Label(self.scrollable_frame, text="SKU Suggestions:", font=(
                'TkDefaultFont', 10, 'bold')).pack(anchor="nw", pady=(0, 5))

            for part_info in self.processed_parts:
                original_desc = part_info["original"]
                suggestions_list = self.current_suggestions.get(
                    original_desc, [])

                part_frame = ttk.Frame(self.scrollable_frame)
                part_frame.pack(pady=5, padx=5, fill="x", anchor="nw")

                ttk.Label(part_frame, text=f"For: '{original_desc}'", font=(
                    'TkDefaultFont', 9, 'bold')).pack(anchor="w")

                if suggestions_list:
                    # Create a StringVar for this part's radio buttons
                    self.selection_vars[original_desc] = tk.StringVar(
                        value=None)  # No default selection

                    for sku, info in suggestions_list:
                        radio_text = f"SKU: {sku} (Conf: {info['confidence']:.3f}, Src: {info['source']})"
                        rb = ttk.Radiobutton(
                            part_frame,
                            text=radio_text,
                            variable=self.selection_vars[original_desc],
                            value=sku  # The value stored when this button is selected
                        )
                        rb.pack(anchor="w", padx=15)
                    # Option to select none
                    rb_none = ttk.Radiobutton(
                        part_frame,
                        text="None of these / Manual Entry",
                        variable=self.selection_vars[original_desc],
                        value=""  # Represent 'no selection' with empty string
                    )
                    rb_none.pack(anchor="w", padx=15)

                else:
                    ttk.Label(part_frame, text="  (No suggestions found)").pack(
                        anchor="w", padx=15)
                ttk.Separator(part_frame, orient='horizontal').pack(
                    fill='x', pady=5)

            # Enable the save button only if there are suggestions
            if any(self.current_suggestions.values()):
                self.save_button.config(state=tk.NORMAL)
            else:
                self.save_button.config(state=tk.DISABLED)

    def save_selections_handler(self):
        """
        Handles the 'Save Confirmed Selections' button click.
        Gathers selected SKUs, adds them to the in-memory Maestro data,
        and writes the updated data back to Maestro.xlsx.
        (Corresponds to Tasks 5.6, 5.7, 5.8)
        """
        global maestro_data_global  # Need to modify the global list
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
                # Only save if a specific SKU was selected (not empty string for 'None')
                if selected_sku:
                    print(
                        f"Selected for '{original_desc}': SKU = {selected_sku}")
                    # Find the corresponding processed part info
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

        # Task 5.7: Add to in-memory Maestro data (avoiding duplicates)
        added_count = 0
        skipped_count = 0

        # Determine next Maestro_ID
        max_id = 0
        if maestro_data_global:
            ids = [entry.get('Maestro_ID', 0) for entry in maestro_data_global if isinstance(
                entry.get('Maestro_ID'), int)]
            if ids:
                max_id = max(ids)
        next_id = max_id + 1

        for selection in selections_to_save:
            # Basic check for duplicates (VIN + Normalized Desc + SKU)
            is_duplicate = False
            for existing_entry in maestro_data_global:
                if (existing_entry.get('VIN_Make') == selection['vin_details'].get('Make') and
                    existing_entry.get('VIN_Model') == selection['vin_details'].get('Model') and
                    # Simple year match for now
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
                    # Store single year for now
                    'VIN_Year_Min': selection['vin_details'].get('Model Year'),
                    # Store single year for now
                    'VIN_Year_Max': selection['vin_details'].get('Model Year'),
                    # Combine Series/Trim if available
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

        # Task 5.8: Write back to Maestro.xlsx
        if added_count > 0:
            print(
                f"Attempting to save {added_count} new entries to {DEFAULT_MAESTRO_PATH}...")
            try:
                # Define columns in the desired order for the Excel file
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
            # This case shouldn't happen if the button was enabled, but good to handle
            messagebox.showinfo(
                "Nothing Saved", "No new confirmations were added.")

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
