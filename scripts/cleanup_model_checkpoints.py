#!/usr/bin/env python3
"""
Standalone script to clean up old model checkpoint files.
This script removes old timestamped model files, keeping only the latest N versions.
"""

import os
import sys
import glob
import argparse
from pathlib import Path

def get_base_path():
    """Get the base path of the project."""
    if hasattr(sys, '_MEIPASS'):
        # Running as PyInstaller bundle
        return os.path.dirname(sys.executable)
    else:
        # Running as script - go up two levels from scripts/
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def cleanup_old_model_checkpoints(model_dir, keep_latest=3, dry_run=False):
    """
    Clean up old model checkpoint files, keeping only the latest N versions.
    
    Args:
        model_dir: Directory containing model files
        keep_latest: Number of latest timestamped models to keep (default: 3)
        dry_run: If True, only show what would be deleted without actually deleting
    """
    if not os.path.exists(model_dir):
        print(f"Error: Model directory not found: {model_dir}")
        return False
    
    try:
        # Find all timestamped model files
        pattern = os.path.join(model_dir, "sku_nn_model_pytorch_optimized_*.pth")
        model_files = glob.glob(pattern)
        
        # Exclude the default model file (without timestamp)
        default_model = os.path.join(model_dir, "sku_nn_model_pytorch_optimized.pth")
        model_files = [f for f in model_files if f != default_model]
        
        if len(model_files) <= keep_latest:
            print(f"Found {len(model_files)} checkpoint files, keeping all (≤ {keep_latest})")
            return True
        
        # Sort by modification time (newest first)
        model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        # Keep the latest N files, delete the rest
        files_to_keep = model_files[:keep_latest]
        files_to_delete = model_files[keep_latest:]
        
        print(f"Found {len(model_files)} checkpoint files in {model_dir}")
        print(f"Keeping latest {keep_latest}:")
        for f in files_to_keep:
            size_mb = os.path.getsize(f) / (1024 * 1024)
            print(f"  ✅ {os.path.basename(f)} ({size_mb:.1f} MB)")
        
        print(f"\n{'Would delete' if dry_run else 'Deleting'} {len(files_to_delete)} old checkpoints:")
        
        deleted_count = 0
        total_size_saved = 0
        for file_path in files_to_delete:
            try:
                file_size = os.path.getsize(file_path)
                size_mb = file_size / (1024 * 1024)
                
                if dry_run:
                    print(f"  🗑️  {os.path.basename(file_path)} ({size_mb:.1f} MB)")
                    deleted_count += 1
                    total_size_saved += file_size
                else:
                    os.remove(file_path)
                    deleted_count += 1
                    total_size_saved += file_size
                    print(f"  🗑️  Deleted: {os.path.basename(file_path)} ({size_mb:.1f} MB)")
                    
            except Exception as e:
                print(f"  ❌ Error processing {os.path.basename(file_path)}: {e}")
        
        if deleted_count > 0:
            total_size_mb = total_size_saved / (1024 * 1024)
            action = "Would save" if dry_run else "Saved"
            print(f"\n✅ {action} {total_size_mb:.1f} MB by {'removing' if not dry_run else 'would remove'} {deleted_count} files")
        
        return True
        
    except Exception as e:
        print(f"Error during cleanup: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Clean up old model checkpoint files")
    parser.add_argument("--keep", type=int, default=3, 
                       help="Number of latest checkpoints to keep (default: 3)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be deleted without actually deleting")
    parser.add_argument("--model-dir", type=str,
                       help="Custom model directory path (default: auto-detect)")
    
    args = parser.parse_args()
    
    # Determine model directory
    if args.model_dir:
        model_dir = args.model_dir
    else:
        base_path = get_base_path()
        model_dir = os.path.join(base_path, "models", "sku_nn")
    
    print("🧹 Model Checkpoint Cleanup Tool")
    print("=" * 40)
    print(f"Model directory: {model_dir}")
    print(f"Keep latest: {args.keep} checkpoints")
    print(f"Mode: {'DRY RUN (no files will be deleted)' if args.dry_run else 'LIVE (files will be deleted)'}")
    print()
    
    if args.dry_run:
        print("⚠️  DRY RUN MODE - No files will actually be deleted")
        print()
    
    success = cleanup_old_model_checkpoints(model_dir, args.keep, args.dry_run)
    
    if success:
        if args.dry_run:
            print("\n💡 To actually delete the files, run without --dry-run")
        else:
            print("\n✅ Cleanup completed successfully!")
    else:
        print("\n❌ Cleanup failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
