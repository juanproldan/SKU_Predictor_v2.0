# 🚀 Quick Reference Guide - Fixacar SKU Predictor v2.1

## 🎯 What's New in v2.1

### ✨ Key Improvements
- **Better Confidence Scores**: 20+ occurrences = 80% confidence
- **Smart Series Matching**: "CX30" automatically becomes "CX-30"
- **Automatic Cleanup**: No more 100+ model files cluttering your disk
- **Faster Database**: Optimized for better performance

## 🔧 Quick Setup

### 1. Run the Application
```bash
python src/main_app.py
```

### 2. Clean Up Old Model Files (One-time)
```bash
# See what would be deleted (safe)
python scripts/cleanup_model_checkpoints.py --dry-run

# Actually clean up (saves ~1GB)
python scripts/cleanup_model_checkpoints.py
```

## 📊 Understanding Confidence Scores

### New Confidence System
| Occurrences | Confidence | Meaning |
|-------------|------------|---------|
| 1 | 30% | ⚠️ Likely error - be cautious |
| 2-4 | 40-45% | 🟡 Low confidence |
| 5-9 | 50-60% | 🟠 Medium-low confidence |
| 10-19 | 60-70% | 🔵 Medium confidence |
| 20+ | 80% | ✅ High confidence - reliable |

### Multi-Source Bonuses
- **Two sources agree**: +10% confidence boost
- **Maestro + Neural Network**: 100% confidence
- **All three sources**: 100% confidence

## 🔄 Series Normalization

### What It Does
Automatically converts series variations to standard format:
- `CX30` → `CX-30`
- `CX 30` → `CX-30`
- `FOCUS ST` → `Focus ST`

### How It Works
1. **During Data Processing**: Handles obvious cases
2. **During Prediction**: Catches remaining variations
3. **Fuzzy Matching**: Fallback for edge cases

### Custom Configuration
Edit `Source_Files/Text_Processing_Rules.xlsx` - Series tab:
```
CX-30    | CX30     | CX 30    | CX_30
Focus ST | FOCUS ST | Focus-ST | FocusST
```

## 🧹 Maintenance

### Automatic Features (No Action Needed)
- ✅ Model cleanup during training
- ✅ Database optimization on startup
- ✅ Series normalization during processing

### Manual Maintenance (Optional)
```bash
# Clean up model files manually
python scripts/cleanup_model_checkpoints.py

# Keep more/fewer checkpoints
python scripts/cleanup_model_checkpoints.py --keep 5

# Check what would be cleaned
python scripts/cleanup_model_checkpoints.py --dry-run
```

## 🎯 Using the Application

### 1. Enter VIN Number
- Application predicts vehicle details automatically
- Series normalization happens behind the scenes

### 2. Enter Part Descriptions
- One description per line
- Application handles text normalization automatically

### 3. Review Results
- **Green confidence (70%+)**: Reliable predictions
- **Yellow confidence (50-69%)**: Moderate reliability
- **Red confidence (<50%)**: Use with caution

### 4. Select SKUs
- Click radio buttons to select preferred SKUs
- Application learns from your selections

## 🔍 Troubleshooting

### Common Issues

#### "Series not normalizing correctly"
1. Check `Text_Processing_Rules.xlsx` - Series tab exists
2. Verify format: first column = canonical form
3. Restart application to reload rules

#### "Too many model files"
```bash
# Check current count
ls models/sku_nn/sku_nn_model_pytorch_optimized_*.pth | wc -l

# Clean up if needed
python scripts/cleanup_model_checkpoints.py
```

#### "Database seems slow"
- Optimizations are applied automatically
- Check if `processed_consolidado.db` exists
- Restart application if needed

#### "Confidence scores seem wrong"
- New system: 20+ occurrences = 80% (this is correct)
- Lower frequencies get lower confidence (this is intentional)
- Single occurrences = 30% (flagged as potential errors)

## 📈 Performance Tips

### For Better Predictions
1. **Use Complete VINs**: More accurate vehicle details
2. **Consistent Descriptions**: Use standard part terminology
3. **Learn from Results**: Select correct SKUs to improve future predictions

### For Better Performance
1. **Regular Cleanup**: Run model cleanup monthly
2. **Update Rules**: Keep Text_Processing_Rules.xlsx current
3. **Monitor Disk Space**: Check models/ directory size

## 🆘 Getting Help

### Log Files
Check `logs/` directory for detailed operation logs

### Debug Information
Application shows detailed processing steps in console

### Configuration Files
- `Source_Files/Text_Processing_Rules.xlsx` - Text processing rules
- `Source_Files/processed_consolidado.db` - Main database
- `models/` - Trained models

## 📋 Checklist for New Users

### Initial Setup
- [ ] Run application to verify it starts
- [ ] Clean up old model files: `python scripts/cleanup_model_checkpoints.py`
- [ ] Verify Text_Processing_Rules.xlsx exists
- [ ] Test with sample VIN and part description

### Regular Usage
- [ ] Enter VIN for vehicle details
- [ ] Add part descriptions (one per line)
- [ ] Review confidence scores
- [ ] Select appropriate SKUs
- [ ] Monitor disk space occasionally

### Maintenance (Monthly)
- [ ] Run model cleanup if training frequently
- [ ] Update series normalization rules if needed
- [ ] Check application logs for any issues
- [ ] Verify database performance is good

## 🔮 What's Coming Next

### Planned Features
- Enhanced caching for even faster predictions
- Machine learning-based series detection
- Automated rule discovery from data
- Performance monitoring dashboard

---

## 📞 Support

For technical issues or questions about the improvements:
1. Check this guide first
2. Review logs in `logs/` directory
3. Test with sample data to isolate issues
4. Document specific error messages for troubleshooting
