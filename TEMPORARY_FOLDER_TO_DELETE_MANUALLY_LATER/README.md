# 🚗 Fixacar SKU Predictor v2.0

A machine learning-powered application for predicting automotive part SKUs based on VIN numbers and part descriptions.

## 📁 Project Structure

```
010_SKU_Predictor_v2.0/
├── 📱 APPLICATION
│   ├── src/
│   │   ├── main_app.py                    # 🎯 MAIN GUI APPLICATION
│   │   ├── train_vin_predictor.py         # 🧠 VIN prediction model trainer
│   │   ├── train_sku_nn_predictor_pytorch_optimized.py  # 🧠 SKU neural network trainer
│   │   ├── unified_consolidado_processor.py  # 📊 Data processor
│   │   ├── download_consolidado.py        # 📥 Data downloader
│   │   ├── models/                        # 🤖 Model implementations
│   │   ├── utils/                         # 🛠️ Utility functions
│   │   ├── core/                          # 🏗️ Core business logic
│   │   └── gui/                           # 🖼️ GUI components
│   │
├── 📊 DATA
│   ├── Source_Files/
│   │   ├── Consolidado.json               # 📋 Main training data
│   │   ├── Maestro.xlsx                   # 📈 Master data
│   │   ├── Text_Processing_Rules.xlsx     # 📝 Text normalization rules
│   │   └── processed_consolidado.db       # 🗄️ Processed database
│   │
├── 🧠 MODELS
│   ├── models/
│   │   ├── vin_*.joblib                   # 🔍 VIN prediction models
│   │   └── sku_nn/                        # 🧠 Neural network models
│   │
├── 📦 DEPLOYMENT
│   ├── deployment/
│   │   ├── dist/                          # 🚀 Built executables
│   │   ├── fixacar_sku_predictor.spec     # 📋 Main app build spec
│   │   ├── Fixacar_SKU_Trainer.spec      # 📋 Trainer build spec
│   │   ├── build_executables_improved.bat # 🔨 Build script
│   │   ├── create_deployment_package.bat  # 📦 Deployment packager
│   │   ├── setup_consolidado_scheduler.bat # ⏰ Scheduler setup
│   │   └── hooks/                         # 🛠️ PyInstaller hooks
│   │
├── 🔧 AUTOMATION
│   ├── scripts/
│   │   ├── run_full_training.py           # 🎯 Complete training pipeline
│   │   ├── validate_existing_models.py   # ✅ Model validation
│   │   ├── monthly_full_retraining.py    # 📅 Monthly training
│   │   ├── weekly_incremental_training.py # 📅 Weekly training
│   │   └── *.bat                          # 🖥️ Windows automation
│   │
├── ⚡ PERFORMANCE
│   ├── performance_improvements/
│   │   ├── cache/                         # 💾 Caching system
│   │   ├── optimizations/                 # ⚡ Speed optimizations
│   │   ├── enhanced_text_processing/      # 📝 Advanced text processing
│   │   └── validation/                    # ✅ Performance validation
│   │
└── 📚 DOCUMENTATION
    ├── README.md                          # 📖 This file
    └── docs/
        ├── 010_SKU_Predictor_v2.0_PRD.md # 📋 Product requirements
        ├── CLIENT_DEPLOYMENT_GUIDE.md    # 🚀 Deployment guide
        └── DEPLOYMENT_README.md          # 📦 Deployment instructions
```

## 🚀 Quick Start

### Running the Application
```bash
# Navigate to project directory
cd "C:\Users\juanp\Documents\Python\0_Training\017_Fixacar\010_SKU_Predictor_v2.0"

# Run the main application
python src/main_app.py
```

### Building Executables
```bash
# Build all executables
deployment/build_executables_improved.bat

# Create deployment package
deployment/create_deployment_package.bat
```

## 🎯 Main Components

### 1. **Main Application** (`src/main_app.py`)
- GUI interface for SKU prediction
- VIN-based vehicle detail prediction
- Multi-source SKU prediction (Maestro, Neural Network, Database)
- User learning system

### 2. **Training Scripts**
- **VIN Predictor**: `src/train_vin_predictor.py`
- **SKU Neural Network**: `src/train_sku_nn_predictor_pytorch_optimized.py`

### 3. **Data Processing**
- **Consolidado Processor**: `src/unified_consolidado_processor.py`
- **Data Downloader**: `src/download_consolidado.py`

### 4. **Performance Improvements**
- Caching system for faster predictions
- Database optimizations
- Enhanced text processing with equivalencias and abbreviations

## 🔧 Development

### Prerequisites
- Python 3.11+
- Required packages: `pip install -r requirements.txt`
- PyTorch for neural network models
- scikit-learn for traditional ML models

### Training Models
```bash
# Full training pipeline
python scripts/run_full_training.py

# Individual training
python src/train_vin_predictor.py
python src/train_sku_nn_predictor_pytorch_optimized.py
```

## 📦 Deployment

The project includes automated deployment scripts for client installations:
- Standalone executables in `deployment/dist/`
- Automated training schedulers
- Complete deployment packages

See `docs/CLIENT_DEPLOYMENT_GUIDE.md` for detailed deployment instructions.

## 🏗️ Architecture

The application uses a multi-layered architecture:
1. **Presentation Layer**: Tkinter GUI
2. **Business Logic**: Prediction engines and text processing
3. **Data Layer**: SQLite databases and Excel files
4. **ML Layer**: PyTorch neural networks and scikit-learn models

## 📊 Performance Features

- **Caching**: Intelligent prediction caching
- **Parallel Processing**: Multi-threaded predictions
- **Text Optimization**: Advanced Spanish text normalization
- **Database Optimization**: Indexed queries and connection pooling

## 🆕 Recent Improvements (v2.1)

### 🎯 Confidence Scoring Enhancements
- **Database Frequency-Based Confidence**: 20+ occurrences now yield 80% confidence
- **Improved Scaling**: Better confidence distribution for 1-19 occurrences
- **Consensus Logic**: Enhanced multi-source agreement scoring

### 🔄 Series Normalization System
- **Hybrid Approach**: Two-phase normalization (preprocessing + runtime)
- **Phase 1**: Series normalization during Consolidado.json processing
- **Phase 2**: Runtime normalization with fallback to fuzzy matching
- **Configuration**: Series tab in Text_Processing_Rules.xlsx for custom mappings

### 🧹 Model File Management
- **Automatic Cleanup**: Training script now keeps only latest 3 model checkpoints
- **Disk Space Savings**: Prevents accumulation of 100+ checkpoint files (saves ~1GB)
- **Standalone Cleanup Tool**: `scripts/cleanup_model_checkpoints.py` for manual cleanup

### ⚡ Database Connection Optimization
- **WAL Mode**: Write-Ahead Logging for better concurrent access
- **Memory Optimization**: 10MB cache, memory-based temp tables
- **Memory Mapping**: 256MB memory map for faster I/O
- **Connection Management**: Proper cleanup and error handling

### 📝 Text Processing Improvements
- **Unified Loading**: All text processing rules loaded from single Excel file
- **Series Support**: Added series normalization to text processing pipeline
- **Enhanced Logging**: Better debugging output for normalization steps
- **Error Handling**: Graceful fallbacks when rules files are missing

## 🛠️ Configuration Files

### Text_Processing_Rules.xlsx Structure
```
📊 Text_Processing_Rules.xlsx
├── 📋 Equivalencias     # Industry-specific synonyms
├── 📋 Abbreviations     # Common abbreviations and expansions
├── 📋 User_Corrections  # User-guided corrections
└── 📋 Series           # Series normalization mappings
```

### Series Normalization Format
The Series tab supports flexible mapping formats:
- **Simple**: `CX-30 | CX30 | CX 30` (first column is canonical)
- **Maker-Specific**: `MAZDA/CX-30 (DM)/BASICO | MAZDA/CX30 | MAZDA/CX 30`
- **Generic**: `*` prefix for cross-maker mappings

## 🔧 Maintenance Tools

### Model Cleanup
```bash
# Dry run to see what would be deleted
python scripts/cleanup_model_checkpoints.py --dry-run

# Clean up old checkpoints (keep latest 3)
python scripts/cleanup_model_checkpoints.py

# Custom retention
python scripts/cleanup_model_checkpoints.py --keep 5
```

### Database Optimization
The application now automatically applies SQLite optimizations:
- WAL mode for concurrent access
- Increased cache size (10MB)
- Memory-based temporary storage
- Memory mapping for large databases

## 📈 Performance Metrics

### Confidence Scoring Distribution
- **1 occurrence**: 30% confidence (likely errors)
- **2-4 occurrences**: 40-45% confidence
- **5-9 occurrences**: 50-60% confidence
- **10-19 occurrences**: 60-70% confidence
- **20+ occurrences**: 80% confidence (high reliability)

### Series Normalization Coverage
- **Phase 1**: Handles obvious cases during data preprocessing
- **Phase 2**: Runtime normalization for edge cases
- **Fallback**: Existing fuzzy matching for unhandled variations

### Disk Space Optimization
- **Before**: 100+ model checkpoints (~1GB wasted space)
- **After**: 3 latest checkpoints (~25MB total)
- **Savings**: ~975MB per training cycle
