# Fixacar Consolidado.json Downloader

## Overview

The **Fixacar Consolidado Downloader** is a standalone executable that automatically downloads the latest `Consolidado.json` file from the Fixacar S3 bucket. This ensures your SKU Predictor system always has the most up-to-date training data.

## Features

✅ **Automatic Download** - Downloads latest Consolidado.json (221+ MB)
✅ **Progress Tracking** - Real-time download progress with MB indicators
✅ **Direct Overwrite** - Replaces existing file directly (no backup clutter)
✅ **JSON Validation** - Verifies downloaded file integrity (108,340+ records)
✅ **Error Handling** - Comprehensive error handling with detailed logging
✅ **Scheduler Ready** - Perfect for Windows Task Scheduler automation
✅ **Standalone** - No Python installation required (8.8 MB executable)

## Files

| **File** | **Purpose** | **Size** |
|---|---|---|
| `Fixacar_Consolidado_Downloader.exe` | Main executable | 8.8 MB |
| `setup_consolidado_scheduler.bat` | Task Scheduler setup | - |
| `CONSOLIDADO_DOWNLOADER_GUIDE.md` | This documentation | - |

## Manual Usage

### Running the Downloader

1. **Double-click** `Fixacar_Consolidado_Downloader.exe`
2. **Wait** for download to complete (usually 30-60 seconds)
3. **Press Enter** when prompted to exit

### What It Does

1. **Checks existing file** (logs if `Consolidado.json` exists)
2. **Downloads** latest file from: `https://fixacar-public-prod.s3.amazonaws.com/reportes/Consolidado.json`
3. **Validates** JSON structure and record count
4. **Saves** to: `Source_Files\Consolidado.json` (overwrites existing)
5. **Logs** all activity to: `logs\consolidado_download_YYYYMMDD.log`

## Automated Scheduling

### Setup Windows Task Scheduler

1. **Run as Administrator**: `setup_consolidado_scheduler.bat`
2. **Confirm** task creation
3. **Verify** in Task Scheduler (`taskschd.msc`)

### Schedule Details

- **Task Name**: `Fixacar_Consolidado_Download`
- **Frequency**: Daily at 6:00 AM
- **User**: SYSTEM (runs even when logged out)
- **Priority**: Highest

### Manual Task Control

```batch
# Test the task immediately
schtasks /run /tn "Fixacar_Consolidado_Download"

# View task status
schtasks /query /tn "Fixacar_Consolidado_Download"

# Delete the task
schtasks /delete /tn "Fixacar_Consolidado_Download" /f
```

## Logging

### Log Files Location
```
logs/
├── consolidado_download_20250724.log    # Daily download logs
└── scheduler.log                        # Task scheduler logs
```

### Log Format
```
2025-07-24 12:45:09,665 - INFO - [START] Starting Consolidado.json download...
2025-07-24 12:45:09,666 - INFO - [SOURCE] Source: https://fixacar-public-prod.s3.amazonaws.com/reportes/Consolidado.json
2025-07-24 12:45:09,666 - INFO - [TARGET] Target: C:\...\Source_Files\Consolidado.json
2025-07-24 13:51:05,885 - INFO - Backup disabled - overwriting existing file
2025-07-24 12:45:10,734 - INFO - [SIZE] File size: 221.2 MB
2025-07-24 12:45:11,696 - INFO - [PROGRESS] Progress: 4.5% (10.0 MB)
...
2025-07-24 12:45:24,412 - INFO - [SUCCESS] Download completed successfully!
```

## Troubleshooting

### Common Issues

| **Issue** | **Cause** | **Solution** |
|---|---|---|
| **Download timeout** | Slow internet connection | Increase `TIMEOUT_SECONDS` in source |
| **Permission denied** | File in use | Close SKU Predictor before download |
| **Invalid JSON** | Corrupted download | Re-run downloader |
| **Task not running** | Scheduler permissions | Run setup as Administrator |

### Error Messages

- **`[ERROR] Download timeout`** - Network too slow, try again
- **`[ERROR] JSON validation failed`** - File corrupted, re-download
- **`[ERROR] Failed to create backup`** - Check file permissions
- **`[ERROR] Network error`** - Check internet connection

## Integration with SKU Predictor

### Data Flow
```
S3 Bucket → Downloader → Source_Files/Consolidado.json → SKU Trainer → Models
```

### Recommended Schedule
- **Daily Download**: 6:00 AM (fresh data)
- **Weekly SKU Training**: Sunday 7:00 AM (after data update)
- **Monthly Full Retrain**: 1st of month 8:00 AM (complete refresh)

## Configuration

### Customizing Download Location

Edit `TARGET_DIR` in the source code:
```python
TARGET_DIR = r"C:\Your\Custom\Path\Source_Files"
```

### Customizing Schedule

Modify the scheduler setup:
```batch
# Change time (24-hour format)
/st 18:00    # 6:00 PM instead of 6:00 AM

# Change frequency
/sc weekly   # Weekly instead of daily
```

## Security Notes

- **HTTPS Download** - Secure connection to S3
- **File Validation** - JSON structure verification
- **Backup Creation** - Automatic rollback capability
- **System User** - Runs with SYSTEM privileges for reliability

## Support

For issues or questions:
1. **Check logs** in `logs/` directory
2. **Test manually** by running executable
3. **Verify internet** connection to S3
4. **Check permissions** for file access

---

**Last Updated**: 2025-07-24  
**Version**: 1.0  
**Executable Size**: 8.8 MB  
**Python Source**: `src/download_consolidado.py`
