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
