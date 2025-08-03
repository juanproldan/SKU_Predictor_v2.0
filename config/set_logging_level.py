#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unified Logging Level Configuration Script

This script allows users to set the logging level for the entire SKU Predictor project.
It provides an easy way to switch between performance-optimized (minimal logging) 
and verbose debugging modes.

Usage:
    python set_logging_level.py --level NORMAL    # Default production mode
    python set_logging_level.py --level VERBOSE   # Detailed logging
    python set_logging_level.py --level DEBUG     # Full debugging
    python set_logging_level.py --level MINIMAL   # Errors only
    python set_logging_level.py --level SILENT    # Critical errors only

Author: Augment Agent
Date: 2025-01-31
"""

import os
import sys
import argparse
from pathlib import Path

# Available logging levels
LOGGING_LEVELS = {
    'SILENT': 'Only critical errors',
    'MINIMAL': 'Errors and critical information',
    'NORMAL': 'Default production level (warnings and above)',
    'VERBOSE': 'Detailed information (info and above)',
    'DEBUG': 'Full debugging information'
}

def set_environment_variable(level: str):
    """Set the logging level environment variable"""
    
    # Set environment variable for current session
    os.environ['SKU_PREDICTOR_LOG_LEVEL'] = level
    
    # For Windows, also set it persistently (optional)
    if sys.platform == 'win32':
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, 'Environment', 0, winreg.KEY_SET_VALUE)
            winreg.SetValueEx(key, 'SKU_PREDICTOR_LOG_LEVEL', 0, winreg.REG_SZ, level)
            winreg.CloseKey(key)
            print(f"✅ Logging level set persistently in Windows registry: {level}")
        except Exception as e:
            print(f"⚠️ Could not set persistent environment variable: {e}")
            print(f"✅ Logging level set for current session: {level}")
    else:
        print(f"✅ Logging level set for current session: {level}")
        print("💡 To make this persistent, add this to your shell profile:")
        print(f"   export SKU_PREDICTOR_LOG_LEVEL={level}")

def create_logging_config_file(level: str):
    """Create a logging configuration file"""
    
    config_content = f"""# SKU Predictor Logging Configuration
# Generated automatically by set_logging_level.py
# Date: {os.path.basename(__file__)}

# Current logging level: {level}
# Description: {LOGGING_LEVELS.get(level, 'Unknown level')}

# This file is used by the application to determine the logging level
# You can also set the environment variable SKU_PREDICTOR_LOG_LEVEL instead

LOGGING_LEVEL={level}
"""
    
    config_path = Path('logging_config.txt')
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print(f"📄 Created logging configuration file: {config_path}")

def show_current_level():
    """Show the current logging level"""
    
    # Check environment variable first
    env_level = os.getenv('SKU_PREDICTOR_LOG_LEVEL')
    
    # Check config file
    config_path = Path('logging_config.txt')
    file_level = None
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.startswith('LOGGING_LEVEL='):
                        file_level = line.split('=', 1)[1].strip()
                        break
        except Exception:
            pass
    
    print("📊 Current Logging Configuration:")
    print(f"   Environment Variable: {env_level or 'Not set'}")
    print(f"   Configuration File: {file_level or 'Not found'}")
    
    # Determine effective level
    effective_level = env_level or file_level or 'NORMAL'
    print(f"   Effective Level: {effective_level}")
    print(f"   Description: {LOGGING_LEVELS.get(effective_level, 'Unknown level')}")
    
    return effective_level

def show_level_descriptions():
    """Show descriptions of all available logging levels"""
    
    print("📋 Available Logging Levels:")
    print()
    
    for level, description in LOGGING_LEVELS.items():
        print(f"   {level:8} - {description}")
        
        # Add usage examples
        if level == 'SILENT':
            print("             Use for: Production deployment, minimal output")
        elif level == 'MINIMAL':
            print("             Use for: Production monitoring, error tracking")
        elif level == 'NORMAL':
            print("             Use for: Standard operation, balanced performance")
        elif level == 'VERBOSE':
            print("             Use for: Development, progress monitoring")
        elif level == 'DEBUG':
            print("             Use for: Troubleshooting, detailed analysis")
        print()

def test_logging_configuration():
    """Test the current logging configuration"""
    
    print("🧪 Testing logging configuration...")
    
    try:
        # Try to import and use the logging configuration
        sys.path.append('src')
        from utils.logging_config import get_logger, LogLevel
        
        # Create a test logger
        test_logger = get_logger("test_logger")
        
        print("✅ Logging configuration loaded successfully")
        
        # Test different log levels
        print("\n📝 Testing log output at different levels:")
        test_logger.debug("This is a DEBUG message")
        test_logger.info("This is an INFO message")
        test_logger.warning("This is a WARNING message")
        test_logger.error("This is an ERROR message")
        
        print("✅ Logging test completed")
        
    except ImportError as e:
        print(f"⚠️ Could not import logging configuration: {e}")
        print("💡 Make sure you're running this from the project root directory")
    except Exception as e:
        print(f"❌ Error testing logging configuration: {e}")

def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(
        description='Set logging level for SKU Predictor project',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python set_logging_level.py --level NORMAL     # Set to production level
  python set_logging_level.py --level VERBOSE    # Enable detailed logging
  python set_logging_level.py --show             # Show current configuration
  python set_logging_level.py --list             # List all available levels
  python set_logging_level.py --test             # Test current configuration
        """
    )
    
    parser.add_argument('--level', '-l', 
                       choices=list(LOGGING_LEVELS.keys()),
                       help='Set the logging level')
    
    parser.add_argument('--show', '-s', action='store_true',
                       help='Show current logging configuration')
    
    parser.add_argument('--list', action='store_true',
                       help='List all available logging levels')
    
    parser.add_argument('--test', '-t', action='store_true',
                       help='Test the current logging configuration')
    
    args = parser.parse_args()
    
    # Show header
    print("=" * 60)
    print("🔧 SKU Predictor Logging Configuration")
    print("=" * 60)
    
    if args.list:
        show_level_descriptions()
        return
    
    if args.show:
        show_current_level()
        return
    
    if args.test:
        test_logging_configuration()
        return
    
    if args.level:
        print(f"🔄 Setting logging level to: {args.level}")
        print(f"📝 Description: {LOGGING_LEVELS[args.level]}")
        print()
        
        # Set environment variable
        set_environment_variable(args.level)
        
        # Create config file
        create_logging_config_file(args.level)
        
        print()
        print("✅ Logging level configuration complete!")
        print()
        print("💡 Next steps:")
        print("   1. Restart any running applications to apply changes")
        print("   2. Use --test to verify the configuration")
        print("   3. Use --show to check current settings")
        
    else:
        # No arguments provided, show current status and help
        print("📊 Current Status:")
        current_level = show_current_level()
        print()
        print("💡 Usage:")
        print("   Use --level to change the logging level")
        print("   Use --list to see all available levels")
        print("   Use --help for detailed usage information")

if __name__ == "__main__":
    main()
