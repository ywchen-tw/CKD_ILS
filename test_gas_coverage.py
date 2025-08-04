#!/usr/bin/env python3
"""
Test script to demonstrate the improved gas coverage for wavelength ranges.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'sckd'))
from lbl_gas_compute import get_gases_for_wavelength_range, get_gases_from_df, absorption_bands

def test_gas_coverage():
    """Test the gas coverage for various wavelength ranges."""
    
    print("Testing gas coverage for different wavelength ranges:")
    print("=" * 60)
    
    # Test cases with different wavelength ranges
    test_cases = [
        (400, 500),     # Should include O3, NO2, H2O
        (600, 700),     # Should include O2, NO2, H2O
        (300, 800),     # Should include all gases in that range
        (1500, 1600),   # Should include H2O, CO2
        (2000, 2200),   # Should include CO2, CH4
        (620, 625),     # Edge case: small range with overlapping bands
        (344, 448),     # Edge case: exact match with one band
    ]
    
    for wv_start, wv_end in test_cases:
        print(f"\nWavelength range: {wv_start} - {wv_end} nm")
        
        # Old method (only checking endpoints)
        gases_start = get_gases_from_df(wv_start)
        gases_end = get_gases_from_df(wv_end)
        old_method_gases = list(set(gases_start or []) | set(gases_end or []))
        
        # New method (checking entire range)
        new_method_gases = get_gases_for_wavelength_range(wv_start, wv_end)
        
        print(f"  Old method (endpoints only): {sorted(old_method_gases)}")
        print(f"  New method (full range):     {sorted(new_method_gases)}")
        
        # Check if there's a difference
        if set(old_method_gases) != set(new_method_gases):
            missing_gases = set(new_method_gases) - set(old_method_gases)
            print(f"  ⚠️  Missing gases with old method: {sorted(missing_gases)}")
        else:
            print(f"  ✅ Both methods give same result")
    
    print("\n" + "=" * 60)
    print("Summary of absorption bands:")
    for start, end, gases in absorption_bands:
        print(f"  {start:6.1f} - {end:6.1f} nm: {gases}")

if __name__ == "__main__":
    test_gas_coverage()
