#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Monthly Full Retraining Script

This script performs complete monthly retraining:
1. Processes entire Consolidado.json dataset
2. Performs full neural network retraining from scratch
3. Updates all models and encoders
4. Creates comprehensive backups

Designed to run automatically on the first weekend of each month.

Author: Augment Agent
Date: 2024
"""

import os
import sys
import subprocess
import logging
import datetime
import shutil
from pathlib import Path

# Setup logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / "monthly_training.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("MonthlyTraining")

def run_command(command, description, timeout=14400):  # 4 hour default timeout
    """Run a command and log the results."""
    logger.info(f"Starting: {description}")
    logger.info(f"Command: {command}")
    
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True,
            timeout=timeout
        )
        
        if result.returncode == 0:
            logger.info(f"✅ Success: {description}")
            if result.stdout:
                logger.info(f"Output: {result.stdout}")
        else:
            logger.error(f"❌ Failed: {description}")
            logger.error(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"⏰ Timeout: {description} took longer than {timeout/3600:.1f} hours")
        return False
    except Exception as e:
        logger.error(f"💥 Exception in {description}: {e}")
        return False
    
    return True

def backup_models(timestamp):
    """Create comprehensive backup of all models."""
    logger.info("💾 Creating comprehensive model backups...")
    
    backup_dir = Path(f"models/backup/monthly_{timestamp}")
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    # Backup directories to copy
    model_dirs = [
        "models/sku_nn",
        "models"  # VIN models
    ]
    
    try:
        for model_dir in model_dirs:
            if Path(model_dir).exists():
                dest_dir = backup_dir / Path(model_dir).name
                shutil.copytree(model_dir, dest_dir, dirs_exist_ok=True)
                logger.info(f"✅ Backed up {model_dir} to {dest_dir}")
        
        # Also backup the database
        if Path("data/fixacar_history.db").exists():
            shutil.copy2("data/fixacar_history.db", backup_dir / "fixacar_history.db")
            logger.info("✅ Backed up database")
            
        return True
        
    except Exception as e:
        logger.error(f"❌ Backup failed: {e}")
        return False

def main():
    """Main monthly retraining process."""
    start_time = datetime.datetime.now()
    logger.info("=" * 80)
    logger.info("🚀 STARTING MONTHLY FULL RETRAINING")
    logger.info(f"📅 Date: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("⚠️  This process may take 5-14 hours to complete")
    logger.info("=" * 80)
    
    # Change to project root directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    logger.info(f"📁 Working directory: {project_root}")
    
    timestamp = start_time.strftime("%Y%m%d_%H%M%S")
    success = True
    
    # Step 1: Create comprehensive backups
    logger.info("\n💾 Step 1: Creating comprehensive backups...")
    if not backup_models(timestamp):
        logger.warning("⚠️ Backup failed, but continuing with training...")
    
    # Step 2: Process complete dataset
    logger.info("\n📊 Step 2: Processing complete Consolidado.json dataset...")
    logger.info("⏳ This may take 2-6 hours depending on file size...")
    if not run_command(
        "python src/offline_data_processor.py --consolidado Source_Files/Consolidado.json",
        "Processing complete Consolidado.json dataset",
        timeout=21600  # 6 hour timeout
    ):
        success = False
    
    # Step 3: Full NN retraining
    logger.info("\n🧠 Step 3: Full neural network retraining...")
    logger.info("⏳ This may take 3-8 hours depending on dataset size...")
    if not run_command(
        "python src/train_sku_nn_predictor_pytorch_optimized.py --mode full",
        "Full NN retraining from scratch",
        timeout=28800  # 8 hour timeout
    ):
        success = False
    
    # Step 4: Retrain VIN prediction models (if needed)
    logger.info("\n🔍 Step 4: Retraining VIN prediction models...")
    if not run_command(
        "python src/train_vin_predictor.py",
        "VIN prediction model retraining",
        timeout=3600  # 1 hour timeout
    ):
        logger.warning("⚠️ VIN model retraining failed, but continuing...")
    
    # Calculate total time
    end_time = datetime.datetime.now()
    duration = end_time - start_time
    hours, remainder = divmod(duration.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # Final status
    logger.info("\n" + "=" * 80)
    if success:
        logger.info("✅ MONTHLY FULL RETRAINING COMPLETED SUCCESSFULLY")
        logger.info("🎉 All models have been updated with the latest data")
    else:
        logger.error("❌ MONTHLY FULL RETRAINING COMPLETED WITH ERRORS")
        logger.error("🔧 Please check logs and retry failed steps manually")
    
    logger.info(f"⏱️ Total time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    logger.info(f"📅 Completed: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"💾 Backups stored in: models/backup/monthly_{timestamp}")
    logger.info("=" * 80)
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
