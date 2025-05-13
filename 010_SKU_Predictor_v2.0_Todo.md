# Project To-Do List: Fixacar SKU Finder v1.0

This list outlines the development phases and tasks for the Fixacar SKU Finder application, based on the agreed-upon PRD (May 8, 2025).

**Target Environment:** Windows Desktop
**Data Files:**
* `Equivalencias.xlsx` (Read by app)
* `Maestro.xlsx` (Read/Write by app)
* `consolidado.json` (Processed offline by script)
* `fixacar_history.db` (SQLite DB, Queried by app)

---

## Phase 1: Offline Historical Data Processing Script

**Goal:** Build a script that transforms the raw `consolidado.json` and `Equivalencias.xlsx` into a searchable SQLite database (`fixacar_history.db`). This script runs separately from the main application.

* [ ] Task 1.1: Set up a Python environment with necessary libraries (`pandas`, `openpyxl`, `sqlite3`, `json`, `requests` - though `requests` might only be needed in the main app, confirm if script needs API for historical VINs. Let's assume script *doesn't* call API for history for simplicity, uses JSON vehicle data).
* [ ] Task 1.2: Implement Text Normalization function (lowercase, strip whitespace, handle accents, etc. - see PRD 3.3).
* [ ] Task 1.3: Write script logic to read `Equivalencias.xlsx` using `openpyxl` or `pandas`. Build an in-memory dictionary mapping **normalized terms** (from all synonym columns) to their `Equivalencia_Row_ID` (from the `ID` column). Filter for active entries.
* [ ] Task 1.4: Write script logic to read `consolidado.json`. Use a library like `json` to parse it. Iterate through bids and items.
* [ ] Task 1.5: Implement filtering: Only process items that have a `referencia` (SKU) field.
* [ ] Task 1.6: For each valid item, extract `vin_number`, JSON vehicle details (`maker`, `series`, `model`, `fabrication_year`), `original_description`, `sku`.
* [ ] Task 1.7: Apply Text Normalization (Task 1.2) to the `original_description`. Store both original and normalized description.
* [ ] Task 1.8: Look up the **normalized description** in the in-memory Equivalencias map (from Task 1.3) to get the `Equivalencia_Row_ID`. Assign `None` or `-1` if not found.
* [ ] Task 1.9: Implement SQLite database connection and table creation (`fixacar_history.db` with `historical_parts` table schema - see PRD 4.3). Add logic to handle database file creation if it doesn't exist.
* [ ] Task 1.10: Implement loading the processed and linked historical data (VIN, vehicle details, descriptions, SKU, `Equivalencia_Row_ID`) into the `historical_parts` table in `fixacar_history.db`. Consider logic to avoid inserting duplicate historical entries if the script is run multiple times.

**Phase 1 Testing:**

* Run the processing script with a chunk of `consolidado.json` and your `Equivalencias.xlsx`.
* Verify that `fixacar_history.db` file is created.
* Use a SQLite viewer tool (many free ones available online) to open the `.db` file.
* Check that the `historical_parts` table exists with the correct columns.
* Verify that data rows are inserted.
* Check that items without SKUs in the original JSON are *not* in the database.
* Check that descriptions are normalized correctly in the `normalized_description` column.
* Check that the `Equivalencia_Row_ID` column is populated with correct IDs where descriptions match Equivalencias, and `NULL`/`-1` where they don't.

---

## Phase 2: Application Foundation & Data Reading (Main App Startup)

**Goal:** Create the basic application window and implement loading core data files (`Equivalencias.xlsx`, `Maestro.xlsx`) into memory on startup.

* [ ] Task 2.1: Set up your Windows GUI development environment (e.g., install Python, chosen GUI library like `tkinter`, `PyQt6`, `kivy`, and necessary libraries `openpyxl`, `json`, `requests`, `sqlite3`).
* [ ] Task 2.2: Create the basic application window using your chosen GUI library.
* [ ] Task 2.3: Implement application startup logic.
* [ ] Task 2.4: Implement reading `Equivalencias.xlsx` using an Excel library (`openpyxl` recommended for .xlsx). Apply Text Normalization (Task 1.2, should be in a shared module/function). Build the in-memory normalized Equivalencias lookup map (Term -> `Equivalencia_Row_ID`). Handle file not found errors.
* [ ] Task 2.5: Implement reading `Maestro.xlsx` using an Excel library. Implement logic to create the file with headers if it doesn't exist yet. Read existing data into an in-memory data structure (e.g., list of dictionaries or custom objects). Apply Text Normalization (Task 1.2) to description columns and find/store `Equivalencia_Row_ID` for each Maestro entry using the in-memory Equivalencias map (Task 2.4). Handle file not found errors on startup (create empty structure).

**Phase 2 Testing:**

* Run the main application.
* Verify the application window appears.
* Add debug print statements or logging to confirm that data from `Equivalencias.xlsx` and `Maestro.xlsx` is successfully read into the application's memory structures at startup.
* Check console output to verify the in-memory Equivalencias map contains correct normalized terms and IDs.
* Check console output to verify the in-memory Maestro data structure contains the expected data, including normalized descriptions and `Equivalencia_Row_ID`s.

---

## Phase 3: Input Interface & VIN Processing

**Goal:** Implement the user input fields and the VIN decoding logic.

* [ ] Task 3.1: Add text input field for VIN, multi-line text area for Part Descriptions, and a "Find SKUs" button to the GUI layout (from Task 2.2).
* [ ] Task 3.2: Connect a function to the "Find SKUs" button click event.
* [ ] Task 3.3: Inside the function, read the text from the VIN input field.
* [ ] Task 3.4: Implement basic VIN format validation (e.g., check length is 17). Provide user feedback for invalid format.
* [ ] Task 3.5: Implement calling the NHTSA VIN Decoder API (`https://vpic.nhtsa.dot.gov/api/vehicles/decodevin/...`). Use the `requests` library.
* [ ] Task 3.6: Handle the API response. Extract vehicle details (Make, Model, Year, Series, Body Style). Handle API errors (bad response, network issues) and display messages to the user.
* [ ] Task 3.7: Store the extracted VIN details in memory for use in searching.
* [ ] Task 3.8: Read the text from the Part Descriptions text area. Split the text into individual description strings (e.g., by newline).
* [ ] Task 3.9: For each individual description string, apply Text Normalization (Task 1.2).
* [ ] Task 3.10: For each **normalized user description**, look up its `Equivalencia_Row_ID` in the in-memory Equivalencias map (from Task 2.4). Store the original description, normalized description, and `Equivalencia_Row_ID` together for each input part.

**Phase 3 Testing:**

* Run the application.
* Enter valid and invalid VINs. Click "Find SKUs". Verify VIN validation works and API call succeeds/fails as expected with appropriate messages.
* Enter multiple part descriptions in the text area (with variations in caps, spacing, accents). Click "Find SKUs". Add debug prints to verify descriptions are read, split, normalized, and linked to correct `Equivalencia_Row_ID`s (or marked as unlinked) using the in-memory map.

---

## Phase 4: Search Logic & Results

**Goal:** Implement the core search algorithm using loaded data and the SQLite DB, and prepare the results.

* [ ] Task 4.1: In the function triggered by "Find SKUs", establish a connection to the `fixacar_history.db` SQLite file (Task 3.5 established the connection, ensure it's available here).
* [ ] Task 4.2: For each original user description (and its VIN details, normalized description, and `Equivalencia_Row_ID` from Phase 3):
* [ ] Task 4.3: **Implement Maestro Search:** Query the in-memory Maestro data structure (from Task 2.5). Search for entries where VIN details match (Task 3.7) AND the stored `Equivalencia_Row_ID` matches the user input's `Equivalencia_Row_ID` (from Task 3.10). Collect matching `Confirmed_SKU`s with confidence 1.0.
* [ ] Task 4.4: **Implement SQLite Search:** Query the `fixacar_history.db` database (Task 4.1) using SQL. Search the `historical_parts` table (Phase 1) for entries where VIN details match (Task 3.7) AND the stored `Equivalencia_Row_ID` matches the user input's `Equivalencia_Row_ID`. Collect matching `sku`s.
* [ ] Task 4.5: Implement calculating confidence for SQLite results (e.g., simple count/frequency of an SKU for that Equivalencia ID/VIN combination within the query results). Assign confidence < 1.0.
* [ ] Task 4.6: **Implement Fallback Search (if `Equivalencia_Row_ID` is None/-1):** If the user description wasn't found in Equivalencias: Query both in-memory Maestro and `fixacar_history.db` (Task 4.4) by directly comparing the normalized user input description (Task 3.9) to the stored `normalized_description` fields. Collect results with very low confidence (e.g., 0.1-0.4).
* [ ] Task 4.7: Combine search results from Maestro, SQLite (ID match), and Fallback (text match). Remove duplicate SKUs.
* [ ] Task 4.8: Implement sorting the combined list of SKUs by confidence, highest first. Store this list of suggestions associated with the original user description.

**Phase 4 Testing:**

* Ensure Phase 1, 2, 3 are complete.
* Run the application. Enter VINs/Descriptions that:
    * Have matches *only* in Maestro.
    * Have matches *only* in SQLite (via Equivalencia ID).
    * Have matches in *both* Maestro and SQLite.
    * Have the same synonym variations found in Equivalencias, ensuring they map to the same `Equivalencia_Row_ID` and find the same SKUs.
    * Are *not* in Equivalencias but might match historical original descriptions (fallback).
* Use debug prints to verify search functions return expected SKUs with correct confidence and ranking.

---

## Phase 5: Output Interface & Learning Mechanism

**Goal:** Display results to the user and implement the learning loop by updating `Maestro.xlsx`.

* [ ] Task 5.1: In the "Find SKUs" function, after search (Phase 4), display the extracted VIN details (Task 3.6) in the output area.
* [ ] Task 5.2: For each original user description (from Task 3.8), display the original description in the output area.
* [ ] Task 5.3: Underneath each description, display the list of suggested SKUs (from Task 4.8) with their confidence/source.
* [ ] Task 5.4: For each suggested SKU displayed, add a GUI element (e.g., a checkbox or radio button next to it) that the user can select to mark it as correct.
* [ ] Task 5.5: Add a "Save Confirmed Selections" button (or integrate saving directly into the selection mechanism if appropriate for the UI).
* [ ] Task 5.6: Connect a function to the "Save Confirmed Selections" action. Read the user's selections from the GUI elements (Task 5.4).
* [ ] Task 5.7: For each SKU the user marked as correct for a specific original description:
    * Gather the data needed for a new Maestro entry: VIN details (Task 3.7), Original Description Input (Task 3.8), its `Equivalencia_Row_ID` (Task 3.10), the selected `Confirmed_SKU`.
    * Create a new entry object for the in-memory Maestro data structure (from Task 3.5). Assign a unique `Maestro_ID`. Add the gathered data, include the Normalized Description Input (Task 3.9), set Confidence to 1.0, Source to "UserConfirmed", and the current Date.
    * Check if an identical entry (matching VIN details, `Equivalencia_Row_ID`, `Confirmed_SKU`) already exists in the in-memory Maestro data to avoid duplicates before adding.
* [ ] Task 5.8: After processing all selections, write the entire updated in-memory Maestro data structure back to the `Maestro.xlsx` file using an Excel library (`openpyxl`). This involves overwriting the old file. Handle file writing errors.
* [ ] Task 5.9: Provide user feedback confirming selections were saved.

**Phase 5 Testing:**

* Ensure Phases 1-4 are complete.
* Run the application. Perform a search. Verify output displays correctly with selectable SKUs.
* Select several SKUs for different descriptions/VINs, including some that already exist in the original `Maestro.xlsx` and some that don't.
* Click "Save Confirmed Selections".
* Verify the application indicates saving was successful.
* Close the application. Manually open `Maestro.xlsx`. Verify that new rows were added correctly with all required data, including `Equivalencia_Row_ID`, and that no exact duplicates were added.

---

## Phase 6: Refinement, Error Handling & Packaging

**Goal:** Polish the application, add robustness, and prepare it for user distribution on Windows.

* [ ] Task 6.1: Implement comprehensive error handling throughout the application (file not found, read/write errors, JSON parsing errors, SQLite errors, API errors, invalid data formats). Provide clear and user-friendly error messages.
* [ ] Task 6.2: Implement a configuration mechanism (e.g., a simple config file or user settings) for the file paths of `Equivalencias.xlsx`, `Maestro.xlsx`, and `fixacar_history.db`.
* [ ] Task 6.3: Enhance the User Interface (UI/UX) for better usability. This might include clearing input fields after search, displaying loading/searching indicators, improving layout, etc.
* [ ] Task 6.4: Review and refactor the codebase for clarity, readability, and maintainability. Add comments where necessary.
* [ ] Task 6.5: Prepare the application for distribution on Windows. This typically involves creating an executable file that bundles the Python interpreter, libraries, and your code using tools like PyInstaller.
* [ ] Task 6.6: Write brief user instructions on how to run the application, where to place the data files, and how to use the interface.

**Phase 6 Testing:**

* Test the application with missing/renamed data files.
* Test with invalid data within files (e.g., non-numeric ID where expected).
* Test with API connection issues.
* Test saving Maestro data to a read-only location.
* Test the packaged executable on a clean Windows machine (if possible) to ensure all dependencies are bundled correctly.
* Review the user instructions for clarity.

---

This to-do list provides a roadmap from raw data processing to a functional application with learning capabilities, broken down into phases you can tackle sequentially, testing as you go. Let me know how you'd like to approach the first tasks!