# 📋 Fixacar SKU Predictor - Changelog

## [2.1.0] - 2025-07-29

### 🎯 Major Improvements

#### Confidence Scoring System Overhaul
- **Enhanced Database Frequency-Based Confidence**
  - 20+ occurrences now yield 80% confidence (previously capped at lower values)
  - Improved scaling for 1-19 occurrences with better distribution
  - Single occurrences now 30% confidence (flagged as likely errors)
  - Better confidence ranges: 1→30%, 2-4→40-45%, 5-9→50-60%, 10-19→60-70%, 20+→80%

#### Series Normalization System
- **Hybrid Two-Phase Approach**
  - **Phase 1**: Preprocessing normalization during Consolidado.json processing
  - **Phase 2**: Runtime normalization with fallback to existing fuzzy matching
  - Handles variations like "CX30" → "CX-30", "CX 30" → "CX-30"
  
- **Configuration System**
  - Added Series tab to Text_Processing_Rules.xlsx
  - Supports maker-specific mappings: `MAZDA/CX-30 (DM)/BASICO`
  - Supports generic mappings: `*` prefix for cross-maker rules
  - Flexible format: first column is canonical form

- **Implementation Details**
  - `normalize_series()` function in main_app.py
  - `normalize_series_preprocessing()` in unified_consolidado_processor.py
  - Integrated into all prediction sources (Maestro, Neural Network, Database)
  - Comprehensive logging for debugging normalization steps

#### Model File Management System
- **Automatic Cleanup During Training**
  - Training script now automatically keeps only latest 3 model checkpoints
  - Prevents accumulation of 100+ checkpoint files
  - Saves approximately 1GB of disk space per training cycle
  
- **Standalone Cleanup Tool**
  - New script: `scripts/cleanup_model_checkpoints.py`
  - Supports dry-run mode to preview deletions
  - Configurable retention count (default: 3)
  - Detailed reporting of space savings

#### Database Connection Optimization
- **SQLite Performance Enhancements**
  - WAL (Write-Ahead Logging) mode for better concurrent access
  - Increased cache size to 10MB for faster queries
  - Memory-based temporary storage for better performance
  - 256MB memory mapping for large database files
  
- **Connection Management**
  - Proper error handling and cleanup
  - Optimized connection lifecycle
  - Better resource management

### 🔧 Technical Improvements

#### Text Processing Infrastructure
- **Unified Loading System**
  - All text processing rules now loaded from single Excel file
  - Added series normalization to text processing pipeline
  - Enhanced error handling with graceful fallbacks
  - Improved logging for debugging text transformations

#### Code Quality Enhancements
- **Better Error Handling**
  - Graceful fallbacks when configuration files are missing
  - Improved exception handling in database operations
  - Better user feedback for missing dependencies

- **Performance Optimizations**
  - Reduced redundant database connections
  - Optimized SQLite pragma settings
  - Better memory management for large datasets

### 📊 Impact Metrics

#### Disk Space Optimization
- **Before**: 148 model checkpoint files (~1.2GB)
- **After**: 3 latest checkpoints (~25MB)
- **Space Saved**: ~1.175GB (97% reduction)

#### Confidence Scoring Accuracy
- **20+ Occurrences**: Now properly weighted at 80% confidence
- **Low Frequency Items**: Better flagged as potentially unreliable
- **Multi-Source Consensus**: Enhanced agreement detection

#### Series Normalization Coverage
- **Preprocessing Phase**: Handles obvious cases during data processing
- **Runtime Phase**: Catches edge cases during prediction
- **Fallback System**: Existing fuzzy matching for unhandled variations

### 🛠️ New Tools and Scripts

#### Model Management
- `scripts/cleanup_model_checkpoints.py` - Standalone cleanup tool
- Integrated cleanup in training script
- Configurable retention policies

#### Configuration Management
- Enhanced Text_Processing_Rules.xlsx with Series tab
- Flexible mapping format for series normalization
- Support for maker-specific and generic rules

### 📝 Documentation Updates

#### README.md Enhancements
- Added "Recent Improvements" section
- Detailed configuration file documentation
- Performance metrics and impact data
- Maintenance tool usage instructions

#### New Documentation
- CHANGELOG.md (this file) for tracking improvements
- Enhanced inline code documentation
- Better error message descriptions

### 🔄 Migration Notes

#### For Existing Installations
1. **Text Processing Rules**: Add Series tab to Text_Processing_Rules.xlsx if custom series normalization is needed
2. **Model Cleanup**: Run `scripts/cleanup_model_checkpoints.py` to clean up existing checkpoint files
3. **Database**: No migration needed - optimizations are applied automatically

#### For New Installations
- All improvements are included by default
- No additional configuration required
- Automatic cleanup and optimization enabled

### 🐛 Bug Fixes

#### Database Operations
- Fixed potential connection leaks in error scenarios
- Improved error handling for missing database files
- Better cleanup in exception cases

#### Text Processing
- Fixed edge cases in series normalization
- Improved handling of empty or null series values
- Better fallback behavior when normalization rules are missing

### ⚡ Performance Improvements

#### Database Query Performance
- WAL mode reduces lock contention
- Larger cache improves query response times
- Memory mapping reduces I/O overhead

#### Model Training Efficiency
- Automatic cleanup prevents disk space issues
- Better resource management during training
- Reduced storage requirements for model artifacts

### 🔮 Future Enhancements

#### Planned Improvements
- Enhanced caching system for series normalization
- Machine learning-based series variation detection
- Automated series mapping discovery from data
- Performance monitoring and metrics collection

---

## [2.0.0] - Previous Release
- Initial release with VIN prediction
- Multi-source SKU prediction system
- Neural network integration
- Database-driven predictions
- User learning system
