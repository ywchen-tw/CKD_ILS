"""
by Yu-Wen Chen (Yu-Wen.Chen@colorado.edu)

This code is a Python adaptation of the MT_CKD water vapor continuum model
originally written in Fortran.

Modified from:
  - Code repository: https://https://github.com/AER-RC/LBLRTM/blob/master/src/contnm.f90 (v12.17 release); accessed on 2025-5-29

"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d



### linear interpolation use the scipy.interpolate.interp1d
def linear_interp(x, y, x_new):
    f = interp1d(x, y, kind='linear', bounds_error=False, fill_value=0)
    return f(x_new)


def radfn_cal(vi, xkt):
    """
    Calculate the radiation term for the line shape.

    Parameters
    ----------
    vi : float
        The line center wavenumber (VI in Fortran).
    xkt : float
        The temperature scaling factor (XKT in Fortran).

    Returns
    -------
    float
        The radiation term RADFN.
    """
    # if xkt > 0.0:
    #     xviox = vi / xkt
    #     # Small-argument limit
    #     if xviox <= 0.01:
    #         return 0.5 * xviox * vi
    #     # General case
    #     elif xviox <= 10.0:
    #         expv = np.exp(-xviox)
    #         return vi * (1.0 - expv) / (1.0 + expv)
    #     # Large-argument limit
    #     else:
    #         return vi
    # else:
    #     # If XKT ≤ 0, no radiation correction
    #     return vi
    xkt_negative = xkt <= 0.0
    xviox = vi / xkt
    xviox_small = xviox <= 0.01
    xviox_mid = (xviox > 0.01) & (xviox <= 10.0)
    xviox_large = xviox > 10.0
    vi_out = np.zeros_like(vi)
    vi_out[xkt_negative] = vi[xkt_negative]
    vi_out[xviox_small] = 0.5 * xviox[xviox_small] * vi[xviox_small]
    expv = np.exp(-xviox[xviox_mid])
    vi_out[xviox_mid] = vi[xviox_mid] * (1.0 - expv) / (1.0 + expv)
    vi_out[xviox_large] = vi[xviox_large]
    return vi_out
    

def pre_xint(cont_v1, cont_v2, cont_dv, npt_cont, abs_v1, abs_v2, abs_dv, npt_abs):
    """
    Compute interpolation indices and fractional weights to map a continuum grid
    onto an absorption grid, using both lower and upper wavenumber bounds.

    For each continuum grid point j (with wavenumber ν_j),
    this function finds the two nearest indices in the absorption grid
    that bracket ν_j, and computes the fractional distance between them
    for linear interpolation.

    Parameters:
    -----------
    cont_v1 : float
        Lower wavenumber of the continuum grid (cm⁻¹).
    cont_v2 : float
        Upper wavenumber of the continuum grid (cm⁻¹).
    cont_dv : float
        Spacing of the continuum grid (cm⁻¹).
    npt_cont : int
        Number of points in the continuum grid.
    abs_v1 : float
        Lower wavenumber of the absorption grid (cm⁻¹).
    abs_v2 : float
        Upper wavenumber of the absorption grid (cm⁻¹).
    abs_dv : float
        Spacing of the absorption grid (cm⁻¹).
    npt_abs : int
        Number of points in the absorption grid.

    Returns:
    --------
    i_lo : ndarray of shape (npt_cont,)
        Index of the lower absorption grid point that brackets each continuum ν_j.
    i_hi : ndarray of shape (npt_cont,)
        Index of the upper absorption grid point that brackets each continuum ν_j.
    frac : ndarray of shape (npt_cont,)
        Fractional distance of each continuum ν_j between abs_nus[i_lo] and abs_nus[i_hi],
        used for linear interpolation (0 ≤ frac ≤ 1).

    Notes:
    ------
    - Continuum wavenumbers are generated as a uniform grid from cont_v1 to cont_v2.
    - Absorption grid wavenumbers are generated as a uniform grid from abs_v1 to abs_v2.
    - If a continuum ν_j lies outside the absorption grid bounds, frac is clamped to [0,1].
    """
    # Continuum wavenumbers (uniformly spaced between cont_v1 and cont_v2)
    cont_nus = np.linspace(cont_v1, cont_v2, npt_cont)
    # Absorption grid wavenumbers (uniformly spaced between abs_v1 and abs_v2)
    abs_nus = np.linspace(abs_v1, abs_v2, npt_abs)

    # For each continuum wavenumber, find insertion index in abs_nus
    idx = np.searchsorted(abs_nus, cont_nus)
    # Clip insertion index to valid range [1, npt_abs-1]
    i_hi = np.clip(idx, 1, npt_abs - 1)
    # Lower index is one less than insertion index
    i_lo = i_hi - 1

    # Compute fractional distances for linear interpolation
    abs_lo = abs_nus[i_lo]
    abs_hi = abs_nus[i_hi]
    frac = (cont_nus - abs_lo) / (abs_hi - abs_lo)
    np.clip(frac, 0.0, 1.0, out=frac)

    return i_lo, i_hi, frac



def xint(cont_values, i_lo, i_hi, frac, abs_array):
    """
    Scatter (add) continuum values onto a global absorption grid using linear interpolation.

    Given an array cont_values of length npt_cont, and precomputed indices
    i_lo, i_hi, and fractional weights frac for each continuum point,
    this function adds the continuum contribution to the absorption array
    abs_array (length npt_abs) by splitting each cont_values[j] between
    abs_array[i_lo[j]] and abs_array[i_hi[j]].

    Parameters:
    -----------
    cont_values : ndarray of shape (npt_cont,)
        Continuum values to be added at each continuum grid point.
    i_lo : ndarray of shape (npt_cont,)
        Lower absorption grid indices bracketing each continuum point.
    i_hi : ndarray of shape (npt_cont,)
        Upper absorption grid indices bracketing each continuum point.
    frac : ndarray of shape (npt_cont,)
        Fractional distances for interpolation (0 ≤ frac ≤ 1).
    abs_array : ndarray of shape (npt_abs,)
        The absorption grid array to which continuum contributions are added.
        Must be mutable and will be modified in place.

    Returns:
    --------
    None

    Notes:
    ------
    - It uses numpy.add.at to handle possible repeated indices robustly.
    - The contribution to abs_array is:
        abs_array[i_lo[j]] += cont_values[j] * (1 - frac[j])
        abs_array[i_hi[j]] += cont_values[j] * frac[j]
      for each j in 0..npt_cont-1.
    """
    # Compute weighted contributions
    weights_lo = cont_values * (1.0 - frac)
    weights_hi = cont_values * frac

    # Add to absorption grid at the two bracketed indices
    np.add.at(abs_array, i_lo, weights_lo)
    np.add.at(abs_array, i_hi, weights_hi)

def read_txt(fname):
    with open(fname, 'r') as f:
        lines = f.readlines()
    data = []
    for line in lines:
        if line.startswith('#'):
            continue
        data.append([float(x) for x in line.split()])
    return np.array(data)


### a function to read dat file
def read_dat(fname):
    with open(fname, 'r') as f:
        lines = f.readlines()
    data = []
    for line in lines:
        if line.startswith('#'):
            continue
        data.append([float(x) for x in line.split()])
    return np.array(data)



def read_hitran_par(fname):
    

    # 1. Define column names in the order listed
    col_names = [
        "molec_id",
        "local_iso_id",
        "nu",
        "sw",
        "a",
        "gamma_air",
        "gamma_self",
        "elower",
        "n_air",
        "delta_air",
        "global_upper_quanta",
        "global_lower_quanta",
        "local_upper_quanta",
        "local_lower_quanta",
        "ierr",
        "iref",
        "line_mixing_flag",
        "gp",
        "gpp",
    ]

    # 2. Define field widths (characters) from the C-style format specifiers:
    #    %2d → 2, %1d → 1, %12.6f → 12, %10.3e → 10, etc.
    # col_widths = [2, 1, 12, 10, 10, 5, 5, 10, 4, 8, 15, 15, 15, 15, 1, 2, 1, 7, 7]
    col_widths = [2, 1, 12, 10, 10, 5, 5, 10, 4, 8, 15, 15, 15, 15, 10, 6, 3, 7, 7]

    # 3. Read the file
    df = pd.read_fwf(
        fname,
        widths=col_widths,
        names=col_names,
        dtype={
            "molec_id": int,            # HITRAN integer ID for this molecule in all its isotopologue forms
            "local_iso_id": int,        # Integer ID of a particular Isotopologue, unique only to a given molecule, in order or abundance
            "nu": float,                # Transition wavenumber
            "sw": float,                # Line intensity, multiplied by isotopologue abundance, at T = 296 K
            "a": float,                 # Einstein A-coefficient in s-1
            "gamma_air": float,         # Air-broadened Lorentzian half-width at half-maximum at p = 1 atm and T = 296 K
            "gamma_self": float,        # Self-broadened HWHM at 1 atm pressure and 296 K
            "elower": float,            # Lower-state energy
            "n_air": float,             # Temperature exponent for the air-broadened HWHM
            "delta_air": float,         # Pressure shift induced by air, referred to p=1 atm
            "ierr": str,                # Ordered list of indices corresponding to uncertainty estimates of transition parameters
            "iref": str,                # Ordered list of reference identifiers for transition parameters
            "line_mixing_flag": str,    # A flag indicating the presence of additional data and code relating to line-mixing
            "gp": float,                # Upper state degeneracy
            "gpp": float,               # Lower state degeneracy
        },
        comment="#",      # if there are comment lines
        skip_blank_lines=True,
    )

    # 4. Strip whitespace from all string (quantum-number) columns
    for col in [
        "global_upper_quanta",          # Electronic and vibrational quantum numbers and labels for the upper state of a transition
        "global_lower_quanta",          # Electronic and vibrational quantum numbers and labels for the lower state of a transition
        "local_upper_quanta",           # Rotational, hyperfine and other quantum numbers and labels for the upper state of a transition
        "local_lower_quanta",           # Rotational, hyperfine and other quantum numbers and labels for the lower state of a transition
        "line_mixing_flag",             # A flag indicating the presence of additional data and code relating to line-mixing
    ]:
        df[col] = df[col].str.strip()

    return df
    


if __name__ == '__main__':

    pass
