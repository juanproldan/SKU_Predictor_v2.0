#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Weekly Incremental Training Script

This script performs weekly incremental training updates:
1. Processes new data from the last week
2. Performs incremental neural network training
3. Updates the production model

Designed to run automatically every Sunday night.

Author: Augment Agent
Date: 2024
"""

import os
import sys
import subprocess
import logging
import datetime
from pathlib import Path

# Setup logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / "weekly_training.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("WeeklyTraining")

def run_command(command, description):
    """Run a command and log the results."""
    logger.info(f"Starting: {description}")
    logger.info(f"Command: {command}")
    
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True,
            timeout=7200  # 2 hour timeout
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
        logger.error(f"⏰ Timeout: {description} took longer than 2 hours")
        return False
    except Exception as e:
        logger.error(f"💥 Exception in {description}: {e}")
        return False
    
    return True

def main():
    """Main weekly training process."""
    start_time = datetime.datetime.now()
    logger.info("=" * 60)
    logger.info("🚀 STARTING WEEKLY INCREMENTAL TRAINING")
    logger.info(f"📅 Date: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    
    # Change to project root directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    logger.info(f"📁 Working directory: {project_root}")
    
    success = True
    
    # Step 1: Process new data (if needed)
    # Note: This assumes new Consolidado.json has been downloaded
    logger.info("\n📊 Step 1: Processing new data...")
    if not run_command(
        "python src/offline_data_processor.py --consolidado Source_Files/Consolidado.json",
        "Processing new Consolidado.json data"
    ):
        success = False
    
    # Step 2: Incremental NN training
    logger.info("\n🧠 Step 2: Incremental neural network training...")
    if not run_command(
        "python src/train_sku_nn_predictor_pytorch_optimized.py --mode incremental --days 7",
        "Incremental NN training (7 days)"
    ):
        success = False
    
    # Step 3: Backup old model and deploy new one
    logger.info("\n💾 Step 3: Model backup and deployment...")
    timestamp = start_time.strftime("%Y%m%d_%H%M%S")
    backup_command = f"copy models\\sku_nn\\sku_nn_model_pytorch_optimized.pth models\\sku_nn\\backup\\sku_nn_model_weekly_{timestamp}.pth"
    
    if not run_command(backup_command, "Backing up current model"):
        logger.warning("⚠️ Model backup failed, but continuing...")
    
    # Calculate total time
    end_time = datetime.datetime.now()
    duration = end_time - start_time
    hours, remainder = divmod(duration.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # Final status
    logger.info("\n" + "=" * 60)
    if success:
        logger.info("✅ WEEKLY INCREMENTAL TRAINING COMPLETED SUCCESSFULLY")
    else:
        logger.error("❌ WEEKLY INCREMENTAL TRAINING COMPLETED WITH ERRORS")
    
    logger.info(f"⏱️ Total time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    logger.info(f"📅 Completed: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
