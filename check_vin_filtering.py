#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Check VIN Filtering During Training

Check why the user's VINs were filtered out during training.

Author: Augment Agent
Date: 2025-08-03
"""

import re

def check_vin_filtering():
    """Check why the user's VINs were filtered out."""
    
    print("🔍 CHECKING VIN FILTERING DURING TRAINING")
    print("=" * 60)
    
    # The user's VINs
    test_vins = [
        "KMHSH81XBCU889564",
        "MALA751AAFM098475", 
        "KMHCT41DAEU610396"
    ]
    
    # The regex from training script
    vin_regex = r"^[A-HJ-NPR-Z0-9]{17}$"
    
    print(f"Training regex: {vin_regex}")
    print(f"This regex excludes: I, O, Q characters")
    print()
    
    for vin in test_vins:
        print(f"Testing VIN: {vin}")
        
        # Check length
        if len(vin) != 17:
            print(f"  ❌ Wrong length: {len(vin)}")
            continue
        else:
            print(f"  ✅ Length OK: {len(vin)}")
        
        # Check regex
        if re.match(vin_regex, vin):
            print(f"  ✅ Regex match: PASSED")
        else:
            print(f"  ❌ Regex match: FAILED")
            
            # Check which characters are problematic
            allowed_chars = set("ABCDEFGHJKLMNPRSTUVWXYZ0123456789")
            vin_chars = set(vin)
            invalid_chars = vin_chars - allowed_chars
            
            if invalid_chars:
                print(f"     Invalid characters found: {invalid_chars}")
                for i, char in enumerate(vin):
                    if char in invalid_chars:
                        print(f"     Position {i+1}: '{char}' is not allowed")
        
        print()
    
    print("💡 CONCLUSION:")
    print("If any VIN fails the regex check, it gets excluded from training.")
    print("The regex excludes I, O, Q characters to follow VIN standards.")
    print("But some real VINs in your database might contain these characters.")

if __name__ == "__main__":
    check_vin_filtering()
