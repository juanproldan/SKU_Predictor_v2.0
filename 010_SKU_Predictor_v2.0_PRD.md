# Product Requirements Document (PRD)

**Document Title:** Fixacar SKU Finder Application v1.0

**Version:** 1.4 (Updated)

**Date:** May 9, 2025

**Prepared By:** Gemini (Collaborative with User)

**Target Environment:** Windows Desktop

---

## 1. Introduction

* **1.1 Purpose:** This document outlines the requirements for a standalone Windows desktop application for Fixacar employees. Its primary purpose is to streamline the bidding process for collision car parts by **predicting** and suggesting relevant SKUs based on vehicle identification numbers (VINs) and part descriptions. This is achieved by leveraging a robust **machine learning model** trained on historical data, incorporating synonym mapping, and a learning mechanism via user feedback. The system is specifically designed to handle the variability in human-entered part descriptions and to predict SKUs for new vehicles (new VINs) not previously encountered.
* **1.2 Goals:**
    * Enable Fixacar employees (non-technical users) to quickly find likely collision part SKUs through **data-driven prediction**.
    * Improve bid creation efficiency and accuracy, reducing errors caused by varied part descriptions.
    * Utilize existing historical bid data (`consolidado.json` processed offline), a synonym list (`Equivalencias.xlsx`), user feedback (`Maestro.xlsx`), and **VIN decoding (via lookup database/rules)** to generalize predictions to new scenarios.
    * Implement a learning loop via `Maestro.xlsx` to reinforce correct predictions and continuously enhance future suggestions based on expert user validation.
    * Provide a simple and intuitive user interface.
    * Handle the large volume of historical data efficiently for offline model training and fast prediction inference.
* **1.3 Scope (v1.0):** Focuses on core SKU **prediction** and suggestion for collision parts based on specified data sources and the learning mechanism.
* **1.4 Out of Scope (v1.0):** Inventory checking, pricing, order placement, integration with other Fixacar systems beyond specified files, handling non-collision parts unless incidentally present in historical data.

## 2. User Stories

* As a Fixacar employee preparing a bid, I want to open the application and enter the VIN and a list of required part descriptions.
* As a Fixacar employee, I want the application to show me the vehicle's key details (Make, Model, Year, Series) based on the VIN.
* As a Fixacar employee, for each part description I entered, I want the application to **predict** and provide a ranked list of probable SKUs for that vehicle and part, even if the description varies or it's a new vehicle VIN.
* As a Fixacar employee, I want to see how confident the system is about each suggested SKU, including predictions and Maestro matches.
* As a Fixacar employee using my expertise, I want to select the correct SKU(s) from the suggested list or manually enter one.
* As a Fixacar employee, I want my correct selections to be saved automatically to improve future **predictions** and suggestions for similar parts and vehicles by updating the Maestro file.
* As a Fixacar employee, I want the system to learn when a suggested SKU was incorrect based on my feedback (declining or selecting a different one).

## 3. Functional Requirements

* **3.1 Application Interface:**
    * Must be a simple GUI application for Windows.
    * Includes VIN text input, multi-line Part Descriptions text area, and a "Find SKUs" button.
    * Includes a clear output area for vehicle details and SKU suggestions/predictions per description.
* **3.2 VIN Decoding & Vehicle Identification:**
    * Validate 17-character VIN on button click.
    * **Use a comprehensive lookup database/table or rule-based logic** to determine and store primary vehicle details (Make, Year, Series) from the VIN. Model and Body Style are not currently determined through this process. These details serve as crucial features for SKU prediction.
* **3.3 Text Normalization:**
    * Implement a function to normalize text descriptions: convert to **lowercase**, remove **leading/trailing whitespace**, standardize internal whitespace, remove or standardize **punctuation**, handle **accented characters/diacritics** (e.g., convert 'á' to 'a'). This step is critical for consistent input to both the Equivalency linking and the prediction model.
* **3.4 Equivalency Linking (Reading `Equivalencias.xlsx`):**
    * Upon application startup (and in the offline script), load **`Equivalencias.xlsx`** into an in-memory lookup structure.
    * The `Equivalencias.xlsx` file has headers like `Column1`, `Column2`, ..., `ColumnN`. Each row in this file represents a group of synonymous terms.
    * For each row:
        * Assign a unique `Equivalencia_Row_ID` (e.g., based on the 1-based row index).
        * Iterate through all cells in the row.
        * For each non-empty cell, retrieve the term and apply Text Normalization (3.3).
        * Create an in-memory map linking each **normalized term** to the `Equivalencia_Row_ID` generated for that row.
    * For user input descriptions (and historical descriptions processed offline), use this map to find the `Equivalencia_Row_ID` for matching. This ID serves as a standardized, categorical representation for the description input to the prediction model and for Maestro/history lookup. If a term is not found in the map, assign `None` or `-1` as its `Equivalencia_Row_ID`, allowing the prediction model to handle unrecognized terms through other text features or general context.
* **3.5 Data Loading & Connection:**
    * Upon application startup, load **`Maestro.xlsx`** into an in-memory data structure. When loading, apply Text Normalization (3.3) to relevant description columns and find/store the `Equivalencia_Row_ID` for each entry using the in-memory Equivalencias map (3.4).
    * Upon application startup, load the pre-trained **SKU Prediction Model** (which could be a Neural Network, a Tree-Based Model like XGBoost or Random Forest, or another suitable machine learning model) from its designated file(s). Handle model loading errors.
    * *(The connection to `fixacar_history.db` is now primarily for the offline script's use in preparing training data for the model, rather than direct querying by the main application for core suggestions in v1.1).*
* **3.6 Search Logic (Prediction & Suggestion):**
    * For each `Original_Description_Input` from the user:
        * Apply Text Normalization (3.3) to get the normalized input.
        * Use the normalized input to find the `Equivalencia_Row_ID` from the in-memory Equivalencias map (3.4).
        * Keep the normalized user input description for potential model input.
        * **Suggestion Strategy:**
            * **Step 1: Search In-Memory Maestro Data (Highest Confidence/Priority):** Query the in-memory Maestro data structure by matching VIN details (Make, Year, Series) AND the normalized description OR the `Equivalencia_Row_ID`. Prioritize matches found in Maestro. Assign Confidence 1.0. These are user-vetted and take precedence as highly reliable suggestions.
            * **Step 2: Generate Prediction from Model:** If no exact match is found in Maestro, or to provide supplementary suggestions, prepare input features for the loaded **SKU Prediction Model**. These features will include the VIN details (Make, Year, Series) and the processed description (e.g., the `Equivalencia_Row_ID`, or potentially other text features like word embeddings if the model requires it, to handle variations not covered by equivalencies). The model outputs a ranked list of predicted SKUs with associated confidence scores (REAL, e.g., 0.0 to 1.0). This step enables generalization for new VINs and varied descriptions.
            * **Step 3: Rank and Combine:** Merge results from Maestro (if any) and the Prediction Model. Remove duplicate SKUs, prioritizing the Maestro source if an SKU appears in both. Sort the combined list by confidence score (Maestro 1.0, then descending model confidence).
* **3.7 Output Interface:**
    * Display decoded vehicle details.
    * For each user description, display the original description and ranked SKU **suggestions (from Maestro or Prediction Model)** with confidence/source indication.
    * Provide a selection mechanism for user-validated learning (Accept/Decline/Edit).
    * Indicate descriptions not found in Equivalencias (may affect prediction confidence if the model relies heavily on this ID).
* **3.8 Learning Mechanism (Updating `Maestro.xlsx` & Implicit Feedback):**
    * When user selects/accepts SKUs: Gather data (VIN details, Original Description, its `Equivalencia_Row_ID`, Selected SKUs).
    * Add new entries to the in-memory Maestro structure (with unique ID, normalized description, `Equivalencia_Row_ID`, confirmed SKU, confidence 1.0, source "UserConfirmed", date, etc.). Avoid exact duplicates based on VIN details and Normalized Description/Equivalencia ID.
    * **Save:** Write the entire current in-memory Maestro data structure back to the **`Maestro.xlsx` file**.
    * When a user **declines** a suggested SKU or selects a *different* SKU than predicted, this provides implicit negative feedback. This feedback will be logged (e.g., in a separate 'FeedbackLog.txt' file or a dedicated table within `fixacar_history.db`) containing the VIN details, original description, the SKUs that were suggested, and the SKU(s) that were ultimately chosen or explicitly declined. This logged information is crucial for identifying cases where the model's predictions were incorrect or ambiguous. It will be incorporated during **future offline model retraining cycles (Section 3.9)** by allowing developers to analyze misclassifications, refine model features, adjust training parameters, or update the `Equivalencias.xlsx` file to enhance the accuracy of subsequent model versions.
* **3.9 Offline Data Processing Script (Separate Concern):**
    * A separate, non-GUI script to be run periodically on Windows.
    * Reads `Consolidado.json` and `Equivalencias.xlsx`.
    * Applies Text Normalization (3.3).
    * Filters `consolidado.json` items for non-empty SKUs.
    * For each filtered item, finds/assigns the `Equivalencia_Row_ID` using the normalized description and `Equivalencias.xlsx`.
    * Connects to/creates the `fixacar_history.db` SQLite database.
    * **Loads the processed data (VIN details, Normalized Description, Equivalencia ID, SKU) into a table** (`historical_parts`) in the SQLite database. Handles appending new data.
    * **Crucially:** This script is also responsible for preparing the dataset from `fixacar_history.db` and performing the **training of the SKU Prediction Model (Machine Learning Model)**. This involves feature engineering from VIN details and description (e.g., using `Equivalencia_Row_ID` or generating text embeddings), selecting and training a suitable model (e.g., a Neural Network, XGBoost, or Random Forest), and evaluating its performance. The trained model parameters/files are the output of this process, ready to be loaded by the main application.

## 4. Data Requirements

* **4.1 Source and Data Files:**
    * `Consolidado.json`: (Large JSON file) Source for offline processing script. Not read directly by the main application.
    * `Equivalencias.xlsx`: (Excel file) Read by main application and offline script. Location configurable.
    * `Maestro.xlsx`: (Excel file) Read and Written by main application. Location configurable.
    * `fixacar_history.db`: (SQLite database file) Populated by offline script, used by script for model training data. *Not* directly queried by the main application for core suggestions in v1.1, but serves as the source for the prediction model's training data. Location configurable.
    * **Prediction Model File(s):** (e.g., `.h5`, `.pkl`, `.json`, custom format) File(s) containing the trained Machine Learning model parameters and/or structure. Created by the offline script, read by the main application. Location configurable.
* **4.2 In-Memory Data Structures (Used by Main Application):**
    * In-memory Maestro Data: Fields mirror `Maestro.xlsx` columns, including `Maestro_ID`, VIN details, `Original_Description_Input`, `Normalized_Description_Input`, `Equivalencia_Row_ID`, `Confirmed_SKU`, `Confidence` (1.0), `Source` ("UserConfirmed"), `Date_Added`.
    * In-memory Equivalencias Lookup: Map linking **normalized terms** (TEXT) from `Equivalencias.xlsx` cells to a unique `Equivalencia_Row_ID` (INTEGER).
    * **Loaded SKU Prediction Model:** The structure and parameters of the trained Machine Learning model, loaded into memory for inference.
* **4.3 File Schemas:**
    * `Equivalencias.xlsx` Schema:
        * Headers: `Column1`, `Column2`, ..., `ColumnN` (e.g., up to `Column9` or as many as needed).
        * Each row contains a set of synonymous terms. All non-empty cells in a given row are considered synonyms of each other.
        * Each row will be assigned a unique `Equivalencia_Row_ID` by the application/script during processing (e.g., 1-based row index). There is no explicit 'ID' or 'Activo' column in this file.
    * `Maestro.xlsx` Schema: Columns: `Maestro_ID` (Unique ID, e.g., INTEGER), `VIN_Make` (TEXT), `VIN_Model` (TEXT), `VIN_Year_Min` (INTEGER), `VIN_Year_Max` (INTEGER), `VIN_Series_Trim` (TEXT), `VIN_BodyStyle` (TEXT), `Original_Description_Input` (TEXT), `Normalized_Description_Input` (TEXT), `Equivalencia_Row_ID` (INTEGER, can be NULL), `Confirmed_SKU` (TEXT), `Confidence` (REAL, 1.0), `Source` (TEXT), `Date_Added` (DATETIME/TEXT).
    * `fixacar_history.db` Schema (`historical_parts` table):
        * `id` (INTEGER PRIMARY KEY AUTOINCREMENT)
        * `vin_number` (TEXT)
        * `vin_make` (TEXT)
        * `vin_model` (TEXT)
        * `vin_year` (INTEGER)
        * `vin_series` (TEXT)
        * `vin_bodystyle` (TEXT)
        * `original_description` (TEXT)
        * `normalized_description` (TEXT)
        * `sku` (TEXT)
        * `Equivalencia_Row_ID` (INTEGER, can be NULL)
        * `source_bid_id` (TEXT/INTEGER, Optional link to original bid)
    * **Prediction Model File(s):** Specific schema depends on the chosen ML framework/format (e.g., Keras/TensorFlow for Neural Networks, Scikit-learn for Tree-based models). Must contain the trained model weights and configuration necessary for making predictions based on VIN details and description features.
* **4.4 File Locations:** Configurable file paths for `Equivalencias.xlsx`, `Maestro.xlsx`, `fixacar_history.db`, and the Prediction Model file(s).

## 5. Non-Functional Requirements

* **5.1 Performance:**
    * Application startup time may be noticeable due to loading `Equivalencias.xlsx`, `Maestro.xlsx`, and the **Prediction Model** into memory.
    * **SKU Prediction inference time** should be fast (milliseconds) once the model is loaded, as it involves a single inference pass through the trained machine learning model.
    * **Maestro lookup time** will be very fast as it's an in-memory structure.
    * The **Offline Data Processing and Model Training Script's runtime** will depend heavily on the size of `Consolidado.json` and the complexity/size of the chosen machine learning model. Training can be a time-consuming process but does **not** impact the live application's prediction speed.
    * Memory usage for the main application will include the Maestro and Equivalencias data, plus the memory required to load the **Prediction Model**. Model size can vary, but should be manageable for a desktop environment.
* **5.2 Usability:** Interface must be simple and intuitive for non-technical users on Windows.
* **5.3 Reliability:** Handle file access errors (`.xlsx`, `.db`, model files), JSON parsing errors (in script), data validation. Ensure Maestro updates are saved persistently. Handle potential prediction model errors gracefully (e.g., suggest "not found" if confidence is too low or an error occurs).
* **5.4 Maintainability:** Code structure should be clear. Dependencies (Excel libraries, SQLite library, JSON parsing, ML model libraries like TensorFlow/Keras for NNs, scikit-learn/XGBoost for tree models) must be managed.
* **5.5 Environment:** The application is designed specifically for the **Windows** operating system.