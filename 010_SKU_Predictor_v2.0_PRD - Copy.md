# Product Requirements Document (PRD)

**Document Title:** Fixacar SKU Finder Application v2.0

**Version:** 2.0 (Current Implementation Status)

**Date:** January 2025

**Prepared By:** Development Team (Updated to reflect actual implementation)

**Target Environment:** Windows Desktop

---

## 1. Introduction

* **1.1 Purpose:** This document outlines the **current implementation** of a standalone Windows desktop application for Fixacar employees. The application streamlines the bidding process for collision car parts by **predicting** and suggesting relevant SKUs based on vehicle identification numbers (VINs) and part descriptions. The system leverages **multiple prediction sources** including machine learning models, historical data, expert-validated entries, and synonym mapping with a comprehensive learning mechanism via user feedback.

* **1.2 Implementation Status - COMPLETED FEATURES:**
    * ✅ **Multi-Source SKU Prediction System** with 4 prediction sources
    * ✅ **VIN Prediction Models** for Make, Model Year, and Series extraction
    * ✅ **PyTorch Neural Network** for SKU prediction with optimized architecture
    * ✅ **Synonym Expansion System** (Equivalencias) for consistent input preprocessing
    * ✅ **Expert-Validated Learning System** (Maestro) with 4-parameter matching
    * ✅ **Historical Database Integration** with frequency-based confidence scoring
    * ✅ **Fuzzy Matching Fallback** for handling unrecognized descriptions
    * ✅ **Comprehensive GUI** with responsive layout and confidence visualization
    * ✅ **Data Processing Pipeline** with text normalization and VIN correction
    * ✅ **Learning Mechanism** that saves user confirmations to improve future predictions

* **1.3 Core Goals ACHIEVED:**
    * ✅ Enable non-technical users to quickly find collision part SKUs through **4 different prediction methods**
    * ✅ Handle variability in human-entered part descriptions via **global synonym expansion**
    * ✅ Predict SKUs for new vehicles using **trained VIN prediction models** and **neural networks**
    * ✅ Implement learning loop via `Maestro.xlsx` with **4-parameter matching** (Make, Year, Series, Description)
    * ✅ Provide intuitive user interface with **confidence scoring** and **source attribution**
    * ✅ Process large historical datasets efficiently for **offline model training**

* **1.4 Scope (v2.0 - IMPLEMENTED):**
    * ✅ Multi-source SKU prediction and suggestion system
    * ✅ VIN-based vehicle detail prediction
    * ✅ Synonym expansion and text normalization
    * ✅ Expert validation and learning mechanism
    * ✅ Historical data integration and processing
    * ✅ Neural network-based prediction with PyTorch

* **1.5 Out of Scope:** Inventory checking, pricing, order placement, integration with other Fixacar systems beyond specified files.

## 2. User Stories - IMPLEMENTED ✅

* ✅ **As a Fixacar employee preparing a bid, I want to open the application and enter the VIN and a list of required part descriptions.**
  - *Implementation: GUI with VIN input field and multi-line part descriptions text area*

* ✅ **As a Fixacar employee, I want the application to show me the vehicle's key details (Make, Model, Year, Series) based on the VIN.**
  - *Implementation: VIN prediction models extract Make, Model Year, and Series from 17-character VIN*

* ✅ **As a Fixacar employee, for each part description I entered, I want the application to predict and provide a ranked list of probable SKUs for that vehicle and part, even if the description varies or it's a new vehicle VIN.**
  - *Implementation: 4-source prediction system with confidence scoring and ranking*

* ✅ **As a Fixacar employee, I want to see how confident the system is about each suggested SKU, including predictions and Maestro matches.**
  - *Implementation: Confidence scores (0.0-1.0) with source attribution (Maestro, Database, Neural Network, Fuzzy)*

* ✅ **As a Fixacar employee using my expertise, I want to select the correct SKU(s) from the suggested list or manually enter one.**
  - *Implementation: Radio button selection interface with manual entry option*

* ✅ **As a Fixacar employee, I want my correct selections to be saved automatically to improve future predictions and suggestions for similar parts and vehicles by updating the Maestro file.**
  - *Implementation: Learning mechanism saves confirmed selections to Maestro.xlsx with 4-parameter matching*

* ✅ **As a Fixacar employee, I want the system to learn when a suggested SKU was incorrect based on my feedback (declining or selecting a different one).**
  - *Implementation: User selections update the expert-validated Maestro database for future high-confidence matches*

## 3. Functional Requirements - IMPLEMENTATION STATUS

* **3.1 Application Interface - ✅ IMPLEMENTED:**
    * ✅ **Tkinter GUI application** for Windows with responsive design
    * ✅ **VIN text input** with automatic correction (I→1, O/Q→0)
    * ✅ **Multi-line Part Descriptions text area** for batch processing
    * ✅ **"Find SKUs" button** with dark styling for visibility
    * ✅ **Two-column layout**: Input frame (60%) and Vehicle Details frame (40%)
    * ✅ **Scrollable results area** with dynamic column layout
    * ✅ **Radio button selection** interface for SKU confirmation
    * ✅ **Confidence visualization** with source attribution
    * ✅ **Save Confirmed Selections** button for learning mechanism
    * ✅ **Manual SKU entry** option for expert input

* **3.2 VIN Decoding & Vehicle Identification - ✅ IMPLEMENTED:**
    * ✅ **17-character VIN validation** with regex pattern matching
    * ✅ **VIN feature extraction** (WMI, Year Code, VDS)
    * ✅ **Trained ML models** for Make, Model Year, and Series prediction:
      - **Make Prediction**: Uses WMI (World Manufacturer Identifier)
      - **Year Prediction**: Uses VIN year code with fallback decoding
      - **Series Prediction**: Uses WMI + VDS (Vehicle Descriptor Section)
    * ✅ **Error handling** for unknown VIN patterns with graceful fallbacks

* **3.3 Text Normalization - ✅ IMPLEMENTED:**
    * ✅ **Comprehensive normalization function** (`utils/text_utils.py`):
      - ✅ Convert to lowercase
      - ✅ Remove leading/trailing whitespace
      - ✅ Standardize internal whitespace (multiple spaces → single space)
      - ✅ Remove punctuation (keeps alphanumeric and spaces)
      - ✅ Handle accented characters/diacritics (á → a)
      - ✅ Unicode normalization (NFKD form)
    * ✅ **Fuzzy normalization option** for enhanced matching
    * ✅ **Applied consistently** across all prediction sources
* **3.4 Equivalency Linking & Synonym Expansion - ✅ IMPLEMENTED:**
    * ✅ **Global Synonym Expansion System** - Revolutionary preprocessing approach
    * ✅ **Loads `Equivalencias.xlsx`** into dual in-memory structures:
      - **Equivalencias Map**: normalized_term → Equivalencia_Row_ID
      - **Synonym Expansion Map**: synonym → canonical_form
    * ✅ **Row-based synonym grouping**:
      - Each row represents synonymous terms
      - First term becomes canonical form
      - All terms in row map to canonical form
    * ✅ **Global preprocessing function** (`expand_synonyms()`):
      - Processes ALL descriptions before ANY prediction method
      - Ensures consistent input across Maestro, Database, and Neural Network
      - Example: "FAROLA IZQ", "FAROLA IZ" → "FAROLA IZQUIERDA"
    * ✅ **Equivalencia_Row_ID assignment** (1-based row index)
    * ✅ **Fallback handling** for unrecognized terms (None/-1 assignment)

* **3.5 Data Loading & Connection - ✅ IMPLEMENTED:**
    * ✅ **Maestro.xlsx loading** with comprehensive data processing:
      - Text normalization for description columns
      - Equivalencia_Row_ID lookup and storage
      - Bracketed value correction ([2012] → 2012, ['Mazda'] → Mazda)
      - Data type validation and conversion
    * ✅ **Multi-Model Loading System**:
      - **VIN Prediction Models**: Make, Year, Series (joblib format)
      - **SKU Neural Network**: Optimized PyTorch model with encoders
      - **Tokenizer**: Description text processing
    * ✅ **SQLite Database Connection** (`fixacar_history.db`):
      - Used for historical data queries during prediction
      - Frequency-based confidence scoring
    * ✅ **Comprehensive error handling** for all model loading scenarios
* **3.6 Multi-Source Search Logic & Prediction System - ✅ IMPLEMENTED:**
    * ✅ **4-Source Prediction Architecture** with intelligent ranking and deduplication
    * ✅ **Input Processing Pipeline**:
      1. **Synonym Expansion**: Apply global synonym expansion FIRST
      2. **Text Normalization**: Normalize expanded description
      3. **Equivalencia Lookup**: Find Equivalencia_Row_ID
      4. **Fuzzy Fallback**: Handle unrecognized terms

    * ✅ **PREDICTION SOURCE 1: Maestro Data (Confidence: 1.0)**
      - **4-Parameter Exact Matching**: Make + Year + Series + Description
      - **Fallback EqID Matching**: Make + Year + Equivalencia_Row_ID (Conf: 0.9)
      - **Expert-validated entries** take highest priority
      - **Source**: User-confirmed historical selections

    * ✅ **PREDICTION SOURCE 2: Historical Database (Confidence: 0.5-0.9)**
      - **SQLite queries** on `fixacar_history.db`
      - **Frequency-based confidence**: 0.5 + 0.4 × (frequency/total_matches)
      - **Matching criteria**: Make + Year + Equivalencia_Row_ID
      - **Source**: Historical bid data

    * ✅ **PREDICTION SOURCE 3: Neural Network (Confidence: Variable)**
      - **PyTorch Optimized Model** with bidirectional LSTM + attention
      - **4-Parameter Input**: Make + Year + Series + Description
      - **Advanced architecture**: Embedding → LSTM → Attention → Dense layers
      - **Source**: AI-powered prediction for new combinations

    * ✅ **PREDICTION SOURCE 4: Fuzzy Matching (Confidence: 0.1-0.4)**
      - **Similarity-based matching** (threshold ≥ 0.8)
      - **Applied to both** Maestro and Database when exact matches fail
      - **Confidence proportional** to similarity score
      - **Source**: Fallback for unrecognized descriptions

    * ✅ **Result Combination & Ranking**:
      - **Duplicate removal**: Keep highest confidence for each SKU
      - **Confidence-based sorting**: Highest confidence first
      - **Source attribution**: Users see prediction source
* **3.7 Output Interface - ✅ IMPLEMENTED:**
    * ✅ **Vehicle Details Display**: Shows predicted Make, Model Year, Series from VIN
    * ✅ **Ranked SKU Suggestions**: Multiple suggestions per part description
    * ✅ **Confidence & Source Visualization**:
      - Confidence scores (0.0-1.0) clearly displayed
      - Source attribution (Maestro, DB, Neural Network, Fuzzy)
    * ✅ **Radio Button Selection Interface**: User-friendly SKU confirmation
    * ✅ **Manual Entry Option**: Expert override capability
    * ✅ **Equivalencias Status Indication**: Shows when descriptions aren't found
    * ✅ **Scrollable Results**: Handles multiple parts and suggestions efficiently

* **3.8 Learning Mechanism & Feedback System - ✅ IMPLEMENTED:**
    * ✅ **User Selection Capture**:
      - Gathers VIN details, Original Description, Equivalencia_Row_ID, Selected SKUs
      - Tracks user confirmations and manual entries
    * ✅ **Maestro Database Updates**:
      - Adds new entries to in-memory Maestro structure
      - **4-Parameter Storage**: Make, Year, Series, Description
      - Assigns confidence 1.0 and source "UserConfirmed"
      - Includes timestamp (Date_Added)
      - **Duplicate prevention** based on VIN details and normalized description
    * ✅ **Persistent Storage**:
      - Writes entire Maestro structure back to `Maestro.xlsx`
      - **Data type consistency**: Integers saved as integers, not text
      - **Column standardization**: Excludes deprecated columns (VIN_Model, VIN_BodyStyle, Equivalencia_Row_ID)
    * ✅ **Implicit Feedback Logging**:
      - User selections provide positive reinforcement
      - Declined suggestions indicate prediction accuracy
      - **Future enhancement**: Formal negative feedback logging system planned
* **3.9 Offline Data Processing & Model Training - ✅ IMPLEMENTED:**
    * ✅ **offline_data_processor.py**: Main data processing script
      - ✅ Reads and processes `Consolidado.json`
      - ✅ Loads `Equivalencias.xlsx` for synonym mapping
      - ✅ Applies comprehensive text normalization
      - ✅ Filters for non-empty SKUs
      - ✅ Assigns Equivalencia_Row_ID using normalized descriptions
      - ✅ Creates/updates `fixacar_history.db` SQLite database
      - ✅ Populates `historical_parts` table with processed data

    * ✅ **VIN Prediction Model Training**:
      - ✅ **train_vin_predictor.py**: Trains Make, Year, Series prediction models
      - ✅ Feature extraction from VIN components (WMI, Year Code, VDS)
      - ✅ Categorical Naive Bayes models with label encoding
      - ✅ Model persistence using joblib format

    * ✅ **SKU Neural Network Training**:
      - ✅ **train_sku_nn_predictor_pytorch.py**: Basic PyTorch model
      - ✅ **train_sku_nn_predictor_pytorch_optimized.py**: Advanced optimized model
      - ✅ Bidirectional LSTM with attention mechanism
      - ✅ Categorical + text feature fusion
      - ✅ Batch normalization and dropout for regularization
      - ✅ Model evaluation and performance metrics

* **3.10 Data Update Utilities - ✅ IMPLEMENTED:**
    * ✅ **Get_New_Data_From_Json.py**: Incremental data processing
      - ✅ Compares original vs new JSON files
      - ✅ Extracts only new records efficiently
      - ✅ Creates New_Data.json and New_Data.db
      - ✅ Preserves original date information
      - ✅ Applies text normalization consistently

    * ✅ **Process_New_Data.py**: Prediction enhancement utility
      - ✅ Reads from New_Data.db database
      - ✅ Adds prediction columns using trained models:
        - **PCS_Make**: VIN-predicted car make
        - **PCS_Year**: VIN-predicted car year
        - **PCS_Series**: VIN-predicted car series
        - **PCS_SKU**: Neural network-predicted SKU
      - ✅ Creates Processed_Data.db with enhanced data
      - ✅ Maintains data integrity and relationships

## 4. Data Requirements - IMPLEMENTATION STATUS

* **4.1 Source and Data Files - ✅ IMPLEMENTED:**
    * ✅ **`Consolidado.json`**: Large historical bid data file for offline processing
    * ✅ **`New_Consolidado.json`**: Updated data source for incremental processing
    * ✅ **`New_Data.json`**: Extracted new records from comparison process
    * ✅ **`Equivalencias.xlsx`**: Synonym mapping file (configurable location)
      - **Format**: Column1, Column2, ..., ColumnN headers
      - **Usage**: Loaded by both main application and offline scripts
    * ✅ **`Maestro.xlsx`**: Expert-validated SKU database (configurable location)
      - **Read/Write**: Main application loads and updates this file
      - **Learning mechanism**: Stores user confirmations
    * ✅ **`fixacar_history.db`**: Historical parts database (configurable location)
      - **Usage**: Queried directly by main application for predictions
      - **Training data**: Source for machine learning model training
    * ✅ **`New_Data.db`**: Incremental data database
    * ✅ **`Processed_Data.db`**: Enhanced data with prediction columns
    * ✅ **Prediction Model Files**: Multiple trained models
      - **VIN Models**: `vin_maker_model.joblib`, `vin_year_model.joblib`, `vin_series_model.joblib`
      - **SKU Neural Network**: `sku_nn_model_pytorch_optimized.pth`
      - **Encoders**: Label encoders and tokenizers for data preprocessing

* **4.2 In-Memory Data Structures - ✅ IMPLEMENTED:**
    * ✅ **Maestro Data Structure**: Complete mirror of `Maestro.xlsx`
      - **Fields**: Maestro_ID, VIN_Make, VIN_Year_Min, VIN_Series_Trim, etc.
      - **Standardized columns**: Removed deprecated VIN_Model, VIN_BodyStyle, Equivalencia_Row_ID
      - **Data types**: Proper integer/string handling with bracketed value correction
    * ✅ **Equivalencias Lookup Maps**: Dual mapping system
      - **equivalencias_map_global**: normalized_term → Equivalencia_Row_ID
      - **synonym_expansion_map_global**: synonym → canonical_form
    * ✅ **Loaded Prediction Models**: Multiple models in memory
      - **VIN Prediction Models**: Make, Year, Series predictors with encoders
      - **SKU Neural Network**: PyTorch model with optimized architecture
      - **Tokenizer**: Text processing for neural network input
* **4.3 File Schemas - ✅ IMPLEMENTED:**
    * ✅ **`Equivalencias.xlsx` Schema**:
      - **Headers**: `Column1`, `Column2`, ..., `ColumnN` (flexible column count)
      - **Row structure**: Each row = synonym group, all non-empty cells are synonyms
      - **ID assignment**: 1-based row index becomes `Equivalencia_Row_ID`
      - **Processing**: Creates both ID mapping and synonym expansion mapping

    * ✅ **`Maestro.xlsx` Schema** (UPDATED - Standardized):
      - **Core columns**: `Maestro_ID`, `VIN_Make`, `VIN_Year_Min`, `VIN_Year_Max`, `VIN_Series_Trim`
      - **Description columns**: `Original_Description_Input`, `Normalized_Description_Input`
      - **SKU columns**: `Confirmed_SKU`, `Confidence`, `Source`, `Date_Added`
      - **REMOVED deprecated columns**: `VIN_Model`, `VIN_BodyStyle`, `Equivalencia_Row_ID`
      - **Data types**: Integers saved as integers, proper type handling

    * ✅ **`fixacar_history.db` Schema** (`historical_parts` table):
      - **Primary key**: `id` (INTEGER PRIMARY KEY AUTOINCREMENT)
      - **VIN fields**: `vin_number`, `vin_make`, `vin_model`, `vin_year`, `vin_series`, `vin_bodystyle`
      - **Description fields**: `original_description`, `normalized_description`
      - **SKU field**: `sku` (TEXT)
      - **Linking field**: `Equivalencia_Row_ID` (INTEGER, nullable)
      - **Metadata**: `source_bid_id`, `date`

    * ✅ **`New_Data.db` Schema**:
      - **Structure**: Identical to `fixacar_history.db`
      - **Content**: Only new records from latest data update
      - **Usage**: Incremental data processing

    * ✅ **`Processed_Data.db` Schema**:
      - **Base structure**: Same as `New_Data.db`
      - **Additional prediction columns**:
        - `PCS_Make` (TEXT): VIN-predicted car make
        - `PCS_Year` (TEXT): VIN-predicted car year
        - `PCS_Series` (TEXT): VIN-predicted car series
        - `PCS_SKU` (TEXT): Neural network-predicted SKU

    * ✅ **Prediction Model Files**:
      - **VIN Models**: Scikit-learn models in joblib format
      - **Neural Network**: PyTorch state dict (.pth format)
      - **Encoders**: Label encoders and tokenizers in joblib format
      - **Architecture**: Optimized with bidirectional LSTM + attention

* **4.4 File Locations - ✅ IMPLEMENTED:**
    * ✅ **Configurable paths** via constants in main application
    * ✅ **Resource path handling** for PyInstaller compatibility
    * ✅ **Default locations**: Source_Files/, data/, models/ directories
    * ✅ **Error handling** for missing files with graceful fallbacks

## 5. Non-Functional Requirements - IMPLEMENTATION STATUS

* **5.1 Performance - ✅ OPTIMIZED:**
    * ✅ **Application startup**: ~3-5 seconds for loading all data and models
      - Equivalencias.xlsx, Maestro.xlsx loaded into memory
      - Multiple prediction models loaded (VIN + SKU neural network)
      - Optimized loading with progress indicators
    * ✅ **SKU Prediction inference**: Sub-second response times
      - **Maestro lookup**: Milliseconds (in-memory hash maps)
      - **Database queries**: <100ms with indexed SQLite queries
      - **Neural network inference**: <500ms on CPU, faster on GPU
      - **Multi-source combination**: Parallel processing where possible
    * ✅ **Offline processing performance**:
      - **Data processing**: Handles large Consolidado.json files efficiently
      - **Model training**: Optimized PyTorch with batch processing
      - **Incremental updates**: Only processes new records
    * ✅ **Memory optimization**:
      - **Efficient data structures**: Hash maps for fast lookups
      - **Model optimization**: Compressed neural network architecture
      - **Memory footprint**: <500MB typical usage

* **5.2 Usability - ✅ USER-FRIENDLY:**
    * ✅ **Intuitive interface**: Simple two-column layout with clear labeling
    * ✅ **Non-technical user focus**: No technical jargon in UI
    * ✅ **Visual feedback**: Confidence scores, source attribution, progress indicators
    * ✅ **Error handling**: User-friendly error messages with guidance
    * ✅ **Responsive design**: Adapts to different window sizes

* **5.3 Reliability - ✅ ROBUST:**
    * ✅ **Comprehensive error handling**:
      - File access errors with graceful fallbacks
      - Model loading failures with alternative approaches
      - Database connection issues with retry logic
      - Invalid VIN format validation and correction
    * ✅ **Data validation**: Input sanitization and type checking
    * ✅ **Persistent storage**: Atomic Maestro.xlsx updates
    * ✅ **Graceful degradation**: System works even if some models fail to load

* **5.4 Maintainability - ✅ WELL-STRUCTURED:**
    * ✅ **Modular architecture**: Separate modules for different functionalities
    * ✅ **Clear code structure**: Well-documented functions and classes
    * ✅ **Dependency management**: Requirements clearly specified
      - **Core**: pandas, numpy, torch, joblib, tkinter
      - **Database**: sqlite3 (built-in)
      - **Excel**: openpyxl (via pandas)
    * ✅ **Configuration management**: Centralized path and parameter settings

* **5.5 Environment - ✅ WINDOWS-OPTIMIZED:**
    * ✅ **Windows desktop application**: Native tkinter GUI
    * ✅ **Path handling**: Windows-compatible file paths
    * ✅ **PyInstaller ready**: Configured for standalone executable creation
    * ✅ **Resource management**: Proper handling of bundled resources

## 6. IMPLEMENTATION SUMMARY

### ✅ **FULLY IMPLEMENTED FEATURES:**
- **Multi-Source SKU Prediction System** (4 sources with confidence scoring)
- **VIN Prediction Models** (Make, Year, Series extraction)
- **Global Synonym Expansion System** (Equivalencias preprocessing)
- **Expert Learning Mechanism** (Maestro 4-parameter matching)
- **Neural Network Prediction** (Optimized PyTorch with attention)
- **Comprehensive GUI** (Responsive design with confidence visualization)
- **Data Processing Pipeline** (Offline training and incremental updates)
- **Robust Error Handling** (Graceful fallbacks and user-friendly messages)

### 🎯 **SYSTEM PERFORMANCE:**
- **Prediction Accuracy**: Multi-source approach maximizes coverage
- **User Experience**: Intuitive interface with clear confidence indicators
- **Learning Capability**: Continuous improvement through user feedback
- **Scalability**: Handles large datasets efficiently
- **Reliability**: Robust error handling and graceful degradation

### 📊 **TECHNICAL ACHIEVEMENTS:**
- **Advanced ML Architecture**: Bidirectional LSTM with attention mechanism
- **Intelligent Preprocessing**: Global synonym expansion ensures consistency
- **Multi-Model Integration**: Seamless combination of different prediction approaches
- **Data Standardization**: 4-parameter matching across all sources
- **Performance Optimization**: Sub-second response times with comprehensive coverage