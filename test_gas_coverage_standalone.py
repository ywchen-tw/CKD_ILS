#!/usr/bin/env python3
"""
Test script to demonstrate the improved gas coverage for wavelength ranges.
"""

import pandas as pd

# Copy the absorption bands from the main file
absorption_bands = [
    (300.0, 448.0, ["O3"]),
    (344.0, 448.0, ["O3", "NO2"]),
    (448.0, 500.0, ["H2O", "NO2"]),
    (500.0, 620.0, ["H2O", "NO2"]),
    (620.0, 625.0, ["O2", "NO2"]),
    (620.0, 640.0, ["O2",]),
    (640.0, 680.0, ["H2O", "O2"]),
    (680.0, 700.0, ["O2",]),
    (700.0, 750.0, ["H2O", "O2"]),
    (750.0, 760.0, ["O2", ]),
    (760.0, 770.0, ["H2O",  "O2"]),
    (770.0, 780.0, ["O2", ]),
    (780.0, 1240.0, ["H2O"]),         # includes wvl_join symbolically
    (1240.0, 1300.0, ["O2", "CO2"]),
    (1300.0, 1420.0, ["H2O", "CO2"]),
    (1420.0, 1450.0, ["CO2"]),
    (1450.0, 1560.0, ["H2O", "CO2"]),
    (1560.0, 1630.0, ["CO2"]),
    (1630.0, 1940.0, ["H2O"]),
    (1940.0, 2150.0, ["CO2"]),
    (2150.0, 2500.0, ["CH4"])
]

def get_gases_from_df(wavelength):
    """Old method: get gases for a single wavelength point."""
    df = pd.DataFrame(absorption_bands, columns=["start", "end", "gases"])
    match = df[(df["start"] <= wavelength) & (wavelength < df["end"])]
    return match.iloc[0]["gases"] if not match.empty else []

def get_gases_for_wavelength_range(wv_start, wv_end):
    """
    New method: Get all gases that are relevant for any part of the wavelength range [wv_start, wv_end].
    
    Parameters:
    wv_start (float): Start wavelength in nm
    wv_end (float): End wavelength in nm
    
    Returns:
    list: Unique list of all gases needed for the wavelength range
    """
    df = pd.DataFrame(absorption_bands, columns=["start", "end", "gases"])
    
    # Find all bands that overlap with our wavelength range
    # A band overlaps if: band_start < wv_end AND band_end > wv_start
    overlapping_bands = df[
        (df["start"] < wv_end) & (df["end"] > wv_start)
    ]
    
    # Collect all unique gases from overlapping bands
    all_gases = set()
    for _, row in overlapping_bands.iterrows():
        all_gases.update(row["gases"])
    
    return sorted(list(all_gases))

def test_gas_coverage():
    """Test the gas coverage for various wavelength ranges."""
    
    print("Testing gas coverage for different wavelength ranges:")
    print("=" * 70)
    
    # Test cases with different wavelength ranges
    test_cases = [
        (400, 500),     # Should include O3, NO2, H2O
        (600, 700),     # Should include O2, NO2, H2O
        (300, 800),     # Should include all gases in that range
        (1500, 1600),   # Should include H2O, CO2
        (2000, 2200),   # Should include CO2, CH4
        (620, 625),     # Edge case: small range with overlapping bands
        (344, 448),     # Edge case: exact match with one band
        (610, 630),     # This is a critical test case!
    ]
    
    for wv_start, wv_end in test_cases:
        print(f"\nWavelength range: {wv_start} - {wv_end} nm")
        
        # Old method (only checking endpoints)
        gases_start = get_gases_from_df(wv_start)
        gases_end = get_gases_from_df(wv_end)
        old_method_gases = sorted(list(set(gases_start) | set(gases_end)))
        
        # New method (checking entire range)
        new_method_gases = get_gases_for_wavelength_range(wv_start, wv_end)
        
        print(f"  Old method (endpoints only): {old_method_gases}")
        print(f"  New method (full range):     {new_method_gases}")
        
        # Check if there's a difference
        if set(old_method_gases) != set(new_method_gases):
            missing_gases = set(new_method_gases) - set(old_method_gases)
            extra_gases = set(old_method_gases) - set(new_method_gases)
            if missing_gases:
                print(f"  ⚠️  Missing gases with old method: {sorted(missing_gases)}")
            if extra_gases:
                print(f"  ⚠️  Extra gases with old method: {sorted(extra_gases)}")
        else:
            print(f"  ✅ Both methods give same result")
    
    print("\n" + "=" * 70)
    print("Summary of absorption bands:")
    for start, end, gases in absorption_bands:
        print(f"  {start:6.1f} - {end:6.1f} nm: {gases}")
    
    print("\n" + "=" * 70)
    print("Key insight: The old method only checks gases at the START and END wavelengths,")
    print("but misses gases that are only relevant in the MIDDLE of the wavelength range!")
    print("\nFor example, range 610-630 nm:")
    print("- At 610 nm: H2O, NO2 (from 500-620 band)")
    print("- At 630 nm: O2 (from 620-640 band)")
    print("- But there's also a 620-625 band with O2, NO2 that's missed by old method!")

if __name__ == "__main__":
    test_gas_coverage()
