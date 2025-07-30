"""
by Yu-Wen Chen (Yu-Wen.Chen@colorado.edu)

This code is a Python adaptation of the MT_CKD water vapor continuum model
originally written in Fortran.

Modified from:
  - Code repository: https://github.com/AER-RC/MT_CKD_H2O (version 4.3); accessed on 2025-5-29
  - Paper: E.J. Mlawer, K.E. Cady-Pereira, J. Mascio, I.E. Gordon,
    "The inclusion of the MT_CKD water vapor continuum model in the HITRAN molecular spectroscopic database",
    J. Quant. Spectrosc. Radiat. Transfer 306, 108645 (2023)

This implementation reads the MT_CKD absorption coefficients from a NetCDF file,
computes the self and foreign water vapor continuum absorption based on temperature,
pressure, and humidity, and returns them on a user-specified spectral grid using xarray.
"""

import os
import sys
import h5py
import numpy as np
import datetime
import pandas as pd
import time
from scipy.io import readsav
from scipy.interpolate import interp1d

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
from matplotlib import rcParams
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
# mpl.use('Agg')

import netCDF4 as nc
import xarray as xr
import er3t
from hapi import *

db_begin('data')

absorption_bands = [
    (300.0, 448.0, ["O3"]),
    (448.0, 500.0, ["H2O", "O3"]),
    (500.0, 620.0, ["H2O", "O3"]),
    (620.0, 640.0, ["O2", "O3"]),
    (640.0, 680.0, ["H2O", "O3", "O2"]),
    (680.0, 700.0, ["O2", "O3"]),
    (700.0, 750.0, ["H2O", "O3", "O2"]),
    (750.0, 760.0, ["O2", "O3"]),
    (760.0, 770.0, ["H2O", "O3", "O2"]),
    (770.0, 780.0, ["O2", "O3"]),
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


absorption_bands = [
    (300.0, 448.0, ["O3"]),
    (448.0, 500.0, ["H2O",]),
    (500.0, 620.0, ["H2O", ]),
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
    df = pd.DataFrame(absorption_bands, columns=["start", "end", "gases"])
    match = df[(df["start"] <= wavelength) & (wavelength < df["end"])]
    return match.iloc[0]["gases"] if not match.empty else None

# Constants
PI = np.pi
PLANCK = 6.62606876E-27
BOLTZ = 1.3806503E-16
CLIGHT = 2.99792458E+10
AVOGAD = 6.02214199E+23
ALOSMT = 2.6867775E+19
GASCON = 8.314472E+07
RADCN1 = 1.191042722E-12
RADCN2 = 1.4387752
XLOSMT = 2.68675E+19
onepl = 1.001
onemi = 0.999


#  XSELF  H2O self broadened continuum absorption multiplicative factor
XSELF = 1 
#  XFRGN  H2O foreign broadened continuum absorption multiplicative factor
XFRGN = 1 
#  XCO2C  CO2 continuum absorption multiplicative factor
XCO2C = 1 
#  XO3CN  O3 continuum absorption multiplicative factor
XO3CN = 1 
#  XO2CN  O2 continuum absorption multiplicative factor
XO2CN = 1 
#  XN2CN  N2 continuum absorption multiplicative factor
XN2CN = 1 
#  XRAYL  Rayleigh extinction multiplicative factor
XRAYL = 1 


###
WK_CONST = 1.0e-20
# WK_CONST = 1

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

### linear interpolation use the scipy.interpolate.interp1d
def linear_interp(x, y, x_new):
    f = interp1d(x, y, kind='linear', bounds_error=False, fill_value=0)
    return f(x_new)


def read_xarray_file(filename, FRGNX):
    """
    Reads the required data from a NetCDF file using xarray.

    Parameters:
        filename (str): Path to the NetCDF file.
        FRGNX (str): Flag indicating which foreign absorption variable to read ('0' or '1').

    Returns:
        dict: A dictionary containing wavenumbers and absorption data needed for computation.
    """
    ds = xr.open_dataset(filename)
    data = {
        'wavenumber': ds['wavenumbers'].values,
        'for_absco_ref': ds['for_closure_absco_ref' if FRGNX == '1' else 'for_absco_ref'].values,
        'self_absco_ref': ds['self_absco_ref'].values,
        'self_texp': ds['self_texp'].values,
        'ref_temp': ds['ref_temp'].values.item(),
        'ref_press': ds['ref_press'].values.item(),
        'title': ds.attrs.get('Title', '')
    }
    return data

def myradfn(vi, xkt, T):
    """
    Computes the radiation term for the line shape.

    Parameters:
        vi (np.ndarray): Array of wavenumbers (cm⁻¹).
        xkt (float): Ratio of temperature to radiation constant (T / RADCN2).

    Returns:
        np.ndarray: Radiation correction factors for each wavenumber.
    """
    xviokt = vi / xkt
    # print("xviokt:", xviokt)
    rad = np.zeros_like(vi)
    small = xviokt <= 0.01
    mid = (xviokt > 0.01) & (xviokt <= 10)
    rad[small] = 0.5 * xviokt[small] * vi[small]
    expvkt = np.exp(-xviokt[mid])
    rad[mid] = vi[mid] * (1 - expvkt) / (1 + expvkt)
    rad = vi*np.arctanh(PLANCK*CLIGHT/(2*BOLTZ*T))
    return rad

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
    

def pre_xint_h2o(v1ss, v2ss, v1abs, dvabs, nptabs):
    """
    Computes start and end indices for interpolation range.

    Parameters:
        v1ss (float): Start of source wavenumber grid.
        v2ss (float): End of source wavenumber grid.
        v1abs (float): Start of desired wavenumber range.
        dvabs (float): Wavenumber spacing.
        nptabs (int): Number of output points.

    Returns:
        (int, int): ist (start index), lst (end index) for interpolation.
    """
    nbnd_v1c = int(2 + (v1ss - v1abs) / dvabs + 1e-5)
    ist = max(1, nbnd_v1c)
    v1abs_loc = v1abs + dvabs * (ist - 1)
    nbnd_v2c = int(1 + (v2ss - v1abs) / dvabs + 1e-5)
    lst = min(nptabs, nbnd_v2c)
    return ist, lst

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

def myxint(v1a, v2a, dva, a, afact, vft, dvr3, nptabs, n1r3, n2r3):
    """
    Interpolates array `a` from (v1a to v2a) to a new spectral grid.

    Parameters:
        v1a, v2a (float): Bounds of input wavenumber range.
        dva (float): Spacing of the input array.
        a (np.ndarray): Input data to interpolate.
        afact (float): Scaling factor for the interpolation.
        vft (float): First wavenumber of output grid.
        dvr3 (float): Output wavenumber spacing.
        r3 (np.ndarray): Output array to store interpolated values.
        n1r3, n2r3 (int): Index range within the output array to interpolate into.
    """
    recdva = 1. / dva
    r3 = np.zeros(nptabs)
    for i in range(n1r3, n2r3 + 1):
        vi = vft + dvr3 * (i - 1)
        j = int((vi - v1a) * recdva + onepl)
        if j < 2 or j+1 >= len(a):
            continue
        vj = v1a + dva * (j - 1)
        p = recdva * (vi - vj)
        c = (3. - 2. * p) * p * p
        b = 0.5 * p * (1. - p)
        b1 = b * (1. - p)
        b2 = b * p
        conti = -a[j - 2] * b1 + a[j - 1] * (1. - c + b2) + a[j] * (c + b1) - a[j + 1] * b2
        # print(f"Interpolating at index {i}: vi={vi}, j={j}, vj={vj}, p={p}")
        # print(f"Continuum value at index {i}: conti={conti}, a[j-2]={a[j-2]}, a[j-1]={a[j-1]}, a[j]={a[j]}, a[j+1]={a[j+1]}") 
        # print(f"p: {p}, b1: {b1}, b2: {b2}, c: {c}")

        r3[i - 1] += conti * afact
    return r3

def mt_ckd_h2o_absco(p_atm, t_atm, h2o_vmr, nu1abs, nu2abs, dvabs, FRGNX, radflag=True, mt_version=None):
    """
    Main function to compute MT_CKD water vapor continuum absorption coefficients.

    Parameters:
        p_atm (float): Atmospheric pressure in mb.
        t_atm (float): Atmospheric temperature in K.
        h2o_vmr (float): Water vapor volume mixing ratio.
        nu1abs (float): Start wavelength of output range (cm-1).
        nu2abs (float): End wavelength of output range (cm-1).
        dvabs (float): Wavenumber resolution of output grid (cm-1).
        FRGNX (str): Use '1' for closure coefficients, '0' otherwise.
        radflag (bool): Apply Planck radiation correction if True.
        mt_version (str or None): Output variable for version string.

    Returns:
        xarray.Dataset: Contains `self_absco` and `for_absco` on the target wavenumber grid.
    """
    # nm to cm⁻¹
    dat = read_xarray_file("absco-ref_wv-mt-ckd.nc", FRGNX)
    version = dat['title']
    if mt_version is not None:
        mt_version = version[3:50]

    wvn = dat['wavenumber']
    dvc = wvn[1] - wvn[0]
    i = 0
    while wvn[i] <= (nu1abs - 2 * dvc):
        i += 1
    i1 = max(i - 1, 0)
    while i < len(wvn) and wvn[i] < (nu2abs + 2 * dvc):
        i += 1
    i2 = i
    ncoeff = i2 - i1

    # Define some atmospheric parameters
    xkt = t_atm / RADCN2
    # The continuum coefficients stored in the netCDF are valid for a reference density and must be 
    # be scaled by this factor to accout for the given atmospheric density.
    # ref_press (1013 mb) and ref_temp (296K) are read in from absco-ref_wv-mt-ckd.nc
    rho_rat = (p_atm / dat['ref_press']) * (dat['ref_temp'] / t_atm)

    # *****************
    # Compute water vapor self continuum absorption coefficient.

    # Apply temperature dependence to reference water vapor self continuum coefficients
    # and scale to given density.
    sh2o_coeff = dat['self_absco_ref'][i1:i2] * (dat['ref_temp'] / t_atm) ** dat['self_texp'][i1:i2]
    print("max sh2o_coeff:", sh2o_coeff.max(), "min:", sh2o_coeff.min())
    sh2o_coeff *= h2o_vmr * rho_rat
    print("max sh2o_coeff after rho ajustment:", sh2o_coeff.max(), "min:", sh2o_coeff.min())
    
    # Multiply by radiation term if requested
    if radflag:
        rad = myradfn(wvn[i1:i2], xkt, t_atm)
        print("max rad:", rad.max(), "min:", rad.min())
        # plt.plot(wvn[i1:i2], rad, label='Radiation Term')
        # plt.xlabel('Wavenumber (cm⁻¹)')
        # plt.ylabel('Radiation Correction Factor')
        # plt.show()
        sh2o_coeff *= rad
    else:
        rad = np.ones_like(sh2o_coeff)
    # print("max sh2o_coeff after rad ajustment:", sh2o_coeff.max(), "min:", sh2o_coeff.min())

    # Interpolate coefficients to output spectral grid.
    nptabs = int((nu2abs - nu1abs) / dvabs + 1)
    nuout_arr = np.arange(nu1abs, nu2abs + 1e-5, dvabs)
    ist, lst = pre_xint_h2o(wvn[0], wvn[-1], nu1abs, dvabs, nptabs)
    self_absco = myxint(wvn[i1], wvn[i2 - 1], dvc, sh2o_coeff, 1.0, nu1abs, dvabs, nptabs, ist, lst)

    # *****************
    # Compute water vapor foreign continuum absorption coefficient.
    fh2o_coeff = dat['for_absco_ref'][i1:i2] * (1 - h2o_vmr) * rho_rat
    # print("max fh2o_coeff:", fh2o_coeff.max(), "min:", fh2o_coeff.min())
    
    # Multiply by radiation term if requested
    if radflag:
        fh2o_coeff *= rad

    # Interpolate coefficients to output spectral grid.
    for_absco = myxint(wvn[i1], wvn[i2 - 1], dvc, fh2o_coeff, 1.0, nu1abs, dvabs, nptabs, ist, lst)
    # *****************
    
    return self_absco, for_absco, mt_version, wvn[i1:i2], nuout_arr

# Example usage:
# p_atm = 1013.25
# t_atm = 296.0
# h2o_vmr = 0.01
# wv1abs = 1000.0
# wv2abs = 2000.0
# dvabs = 1.0
# FRGNX = '0'
# sh2o, fh2o, mt_ver, wvn = mt_ckd_h2o_absco(p_atm, t_atm, h2o_vmr, wv1abs, wv2abs, dvabs, FRGNX)

def abs_tau_calc(atm0, wv1abs, wv2abs, dvabs,
         slit_wvl, slit_response,
         FRGNX=0, radflag=True, ):
    # Constants equivalent to the Fortran DATA statements
    P0 = 1013.0
    T0 = 296.0
    XLOSMT = 2.68675e19
    
    icflg = 0
    
    
    p_lay = atm0.lay['pressure']['data']
    t_lay = atm0.lay['temperature']['data']
    thickness_lay = atm0.lay['thickness']['data']
    
    # 'air', 'o3', 'o2', 'h2o', 'co2', 'no2'
    h2o_lay = atm0.lay['h2o']['data']
    o3_lay = atm0.lay['o3']['data']
    o2_lay = atm0.lay['o2']['data']
    co2_lay = atm0.lay['co2']['data']
    no2_lay = atm0.lay['no2']['data']
    air_lay = atm0.lay['air']['data']
    n2_lay = air_lay - h2o_lay - o2_lay - co2_lay - no2_lay
    
    
    absrb = np.zeros_like(p_lay)
    
    
    RHOAVE_lay = (p_lay/P0)*(T0/t_lay)
    XKT_lay = t_lay/RADCN2
    amagat_lay = (p_lay/P0)*(273./t_lay)
    
    nu1abs = 1e7 / wv2abs
    nu2abs = 1e7 / wv1abs
    nptabs = int((nu2abs - nu1abs) / dvabs + 1)
    nuout_arr = np.arange(nu1abs, nu2abs + 1e-5, dvabs)


    x_vmr_h2o = h2o_lay/air_lay
    x_vmr_o2  = o2_lay/air_lay
    x_vmr_n2  = n2_lay/air_lay
    x_vmr_co2 = co2_lay/air_lay
    x_vmr_no2 = no2_lay/air_lay
    x_vmr_o3  = o3_lay/air_lay
    
    if icflg > 0:
        x_vmr_n2 = 0
    
        cself = np.zeros_like(p_lay)
        cfor = np.zeros_like(p_lay)
    
    print(nu1abs, nu2abs)
    
    nu_final, cont_tau_final = compute_continuum_profile(
                            nu1abs, nu2abs, dvabs,
                            p_lay, t_lay, thickness_lay, air_lay, 
                            x_vmr_co2, x_vmr_n2, x_vmr_o2, x_vmr_h2o, x_vmr_o3,
                            RHOAVE_lay, XKT_lay, amagat_lay,
                            JRAD=radflag
                            )
    

    lambda_final = 1.0e7/nuout_arr
    
    cont_tau_final_inter = np.zeros((atm0.lay['pressure']['data'].shape[0], nuout_arr.shape[0]))
    for iz in range(atm0.lay['pressure']['data'].shape[0]):
        cont_tau_final_inter[iz, :] = linear_interp(nu_final, cont_tau_final[iz, :], nuout_arr)
    
    print("lambda_final shape:", lambda_final.shape)
    
    # lbl_tau_final = np.zeros((atm0.lay['pressure']['data'].shape[0], lambda_final.shape[0]))
    
    # return nu_final, abs_final, lbl_tau_final
    
    nu_total, lambda_total, lbl_tau_final = lbl_profile(wv1abs, wv2abs, dvabs, atm0)
    
    
    print("lambda_total shape:", lambda_total.shape)
    
    fname_solar = "/Users/yuch8913/programming/er3t/er3t/er3t/data/solar/data/solar_flux/kurudz_full.dat"
    datContent = [i.strip().split() for i in open(fname_solar).readlines()]
    solar_data = np.array(datContent[11:]).astype(np.float32)
    
    solar_lambda_mask = np.logical_and(solar_data[:, 0]>=wv1abs, solar_data[:, 0]<=wv2abs)
    solar_data_interpolate = linear_interp(1.0e7/solar_data[:, 0], solar_data[:, 1], nu_final)
    print("solar_data_interpolate shape:", solar_data_interpolate.shape)
    
    slit_response_fit = interp1d(1.0e7/slit_wvl, slit_response, kind='linear', bounds_error=False, fill_value=0.0)
    slit_response_final = slit_response_fit(nu_final)
    
    tau_final = cont_tau_final_inter + lbl_tau_final
    
    for iz in range(atm0.lay['pressure']['data'].shape[0]):
        print("layer:", iz)
        print("  cont_tau_final_inter max, min:", cont_tau_final_inter[iz, :].max(), cont_tau_final_inter[iz, :].min())
        print("  lbl_tau_final max, min:", lbl_tau_final[iz, :].max(), lbl_tau_final[iz, :].min())
        print("  tau_final max, min:", tau_final[iz, :].max(), tau_final[iz, :].min())
        
    
    # use surface layer
    ind_sort = np.argsort(tau_final[0, :]*slit_response_final)
    solar_tau_sorted = solar_data_interpolate[ind_sort]
    solar_tau_sorted /= 1000 


    return nu_final, cont_tau_final_inter, lbl_tau_final, solar_data_interpolate, slit_response_final, ind_sort

def lbl_profile(wv1abs, wv2abs, abs_dv,
                   atm0):
    abs_gases_1 = get_gases_from_df(wv1abs)
    abs_gases_2 = get_gases_from_df(wv2abs)
    abs_gases = list(set(abs_gases_1) | set(abs_gases_2))
    nu_start = 1e7 / wv2abs
    nu_end = 1e7 / wv1abs
    nu_start_lbl = 1e7 / (wv2abs+5)
    nu_end_lbl = 1e7 / (wv1abs-5)
    if 'H2O' in abs_gases:
        fetch_by_ids('H2O', [1,2,3,4], nu_start_lbl, nu_end_lbl) # H216O, H218O, H217O, HD16O
    if 'CO2' in abs_gases:
        fetch_by_ids('CO2', [7,8,9,10], nu_start_lbl, nu_end_lbl) # 12C16O2, 13C16O2, 16O12C18O, 16O12C17O
    if 'O3' in abs_gases:
        fetch_by_ids('O3', [16,17,18,19,20], nu_start_lbl, nu_end_lbl) # 16O3, 16O16O18O, 16O18O16O, 16O16O17O, 16O17O16O
    if 'CH4' in abs_gases:
        fetch_by_ids('CH4', [32,33,34], nu_start_lbl, nu_end_lbl) # 12CH4, 13CH4, 12CH3D
    if 'O2' in abs_gases:
        fetch_by_ids('O2', [36,37,38], nu_start_lbl, nu_end_lbl) # 16O2, 16O18O, 16O17O
        
    dnu = abs_dv
    
    lay_num = atm0.lay['pressure']['data'].shape[0]
    
    nu_list = np.arange(nu_start, nu_end + 1e-5, dnu)
    wvl_num = nu_list.shape[0]
    print("wvl_num:", wvl_num)
    lambda_total = np.zeros((lay_num, wvl_num))
    nu_total = np.zeros((lay_num, wvl_num))
    coef_h2o_total = np.zeros((lay_num, wvl_num))
    coef_co2_total = np.zeros((lay_num, wvl_num))
    coef_o3_total = np.zeros((lay_num, wvl_num))
    coef_ch4_total = np.zeros((lay_num, wvl_num))
    coef_o2_total = np.zeros((lay_num, wvl_num))
    
    tau_h2o_total = np.zeros((lay_num, wvl_num)) # m-1
    tau_co2_total = np.zeros((lay_num, wvl_num)) # m-1
    tau_o3_total = np.zeros((lay_num, wvl_num)) # m-1
    tau_ch4_total = np.zeros((lay_num, wvl_num)) # m-1
    tau_o2_total = np.zeros((lay_num, wvl_num)) # m-1
    
    dnu = 1
    
    
    for i in range(lay_num):
        T_ = atm0.lay['temperature']['data'][i]
        P_ = (atm0.lay['pressure']['data']/1013.25)[i]
        print(T_, P_)
        if 'H2O' in abs_gases:
            nu_, coef_h2o_ = absorptionCoefficient_Lorentz(SourceTables='H2O', Diluent={'air':1.0}, Environment={'T':T_,'p':P_}, WavenumberStep=abs_dv)
            # coef_h2o_total[i, :] = coef_h2o_.copy()
            coef_h2o_total[i, :] = linear_interp(nu_, coef_h2o_, nu_list)
            tau_h2o_total[i, :] = coef_h2o_total[i, :]*(atm0.lay['h2o']['data'][i])*(atm0.lay['thickness']['data'][i]*1000*100)
            print("h2o number densisty:", atm0.lay['h2o']['data'][i])
            print("nu_:", nu_)
            print("coef_h2o_:", coef_h2o_)
            print("coef_h2o_ max, min:", coef_h2o_.max(), coef_h2o_.min())
            print("coef_h2o_total:", coef_h2o_total[i, :])
            print("coef_h2o_total max, min:", coef_h2o_total.max(), coef_h2o_total.min())
            print("tau_h2o_total max, min:", tau_h2o_total[i, :].max(), tau_h2o_total[i, :].min())
        if 'CO2' in abs_gases:
            nu_, coef_co2_ = absorptionCoefficient_Lorentz(SourceTables='CO2', Diluent={'air':1.0}, Environment={'T':T_,'p':P_}, WavenumberStep=abs_dv)
            coef_co2_total[i, :] = coef_co2_.copy()
            coef_co2_total[i, :] = linear_interp(nu_, coef_co2_, nu_list)
            tau_co2_total[i, :] =  coef_co2_total[i, :]*(atm0.lay['co2']['data'][i])*(atm0.lay['thickness']['data'][i]*1000*100)
            print("tau_co2_total max, min:", tau_co2_total[i, :].max(), tau_co2_total[i, :].min())
        if 'O3' in abs_gases:
            nu_, coef_o3_ = absorptionCoefficient_Lorentz(SourceTables='O3', Diluent={'air':1.0}, Environment={'T':T_,'p':P_}, WavenumberStep=abs_dv)
            # coef_o3_total[i, :] = coef_o3_.copy()
            coef_o3_total[i, :] = linear_interp(nu_, coef_o3_, nu_list)
            tau_o3_total[i, :] = coef_o3_total[i, :]*(atm0.lay['o3']['data'][i])*(atm0.lay['thickness']['data'][i]*1000*100)
            print("tau_o3_total max, min:", tau_o3_total[i, :].max(), tau_o3_total[i, :].min())
        if 'CH4' in abs_gases:
            nu_, coef_ch4_ = absorptionCoefficient_Lorentz(SourceTables='CH4', Diluent={'air':1.0}, Environment={'T':T_,'p':P_}, WavenumberStep=abs_dv)
            # coef_ch4_total[i, :] = coef_ch4_.copy()
            coef_ch4_total[i, :] = linear_interp(nu_, coef_ch4_, nu_list)
            tau_ch4_total[i, :] = coef_ch4_total[i, :]*(atm0.lay['ch4']['data'][i])*(atm0.lay['thickness']['data'][i]*1000*100)
            print("tau_ch4_total max, min:", tau_ch4_total[i, :].max(), tau_ch4_total[i, :].min())
        if 'O2' in abs_gases:
            nu_, coef_o2_ = absorptionCoefficient_Lorentz(SourceTables='O2', Diluent={'air':1.0}, Environment={'T':T_,'p':P_}, WavenumberStep=abs_dv)
            # coef_o2_total[i, :] = coef_o2_.copy()
            coef_o2_total[i, :] = linear_interp(nu_, coef_o2_, nu_list)
            tau_o2_total[i, :] = coef_o2_total[i, :]*(atm0.lay['o2']['data'][i])*(atm0.lay['thickness']['data'][i]*1000*100)
            print("tau_o2_total max, min:", tau_o2_total[i, :].max(), tau_o2_total[i, :].min())
        
    #     _, coef_o3_ = absorptionCoefficient_Lorentz(SourceTables='O3', Diluent={'air':1.0}, Environment={'T':T_,'p':P_})
        lambda_total[i, :] = 10000/nu_list*1000 # in nm
        nu_total[i, :] = nu_list
        
    #     coef_o3_total[i, :] = coef_o3_.copy()
    # sys.exit()
    #     tau_o3_total[i, :] = coef_o3_.copy()*(atm0.lay['o3']['data'][i])*(atm0.lay['thickness']['data'][i]*1000*100)/dnu
    tau_final = tau_h2o_total + tau_co2_total + tau_o3_total + tau_ch4_total + tau_o2_total
    
    return nu_total, lambda_total, tau_final
    
        

def water_continuum_profile(abs_v1, abs_v2, abs_dv,
                   p_arr, T_arr, thickness_arr, air_arr, xH2O_arr, 
                   JRAD=False):
    
    nlev = p_arr.size
    npt_abs = int((abs_v2 - abs_v1) / abs_dv) + 1
    C_h2o_all = np.zeros((nlev, npt_abs), dtype=float)

    for i in range(nlev):
        ds = mt_ckd_h2o_absco(
                            p_atm=p_arr[i],
                            t_atm=T_arr[i],
                            h2o_vmr=xH2O_arr[i],
                            nu1abs=abs_v1,
                            nu2abs=abs_v2,
                            dvabs=abs_dv,
                            FRGNX='0',
                            radflag=JRAD,
                        )
        self_absco = ds[0]
        for_absco = ds[1]
        
        WKH2O = air_arr[i] * xH2O_arr[i]
        WKH2O_col = WKH2O * thickness_arr[i] * 1000 * 100  # in molecules/cm²
        C_h2o_all[i, :] = (self_absco + for_absco) * WKH2O_col
        
    return C_h2o_all
    


def frnco2_scalar(V1ABS, V2ABS,
                  tave, air, thickness, xco2,
                  RHOAVE, JRAD, XKT):
    """
    Compute the CO₂ foreign continuum for a single atmospheric layer.
    
    **mt_ckd_2.0      11 July 2007    sac
                 This continuum differs from mt_ck_1.3 in that an entirely
                 new co2 continuum has been developed based on the line
                 coupling parameters from Hartmann's group as distributed
                 with hitran.  This continuum must be used with lblrtm_v11
                 and spectral line parameters including Hartmann's line
                 parameters for co2.
                 Based on recent validation studies, a scaling of the
                 continuum for v3 is required to achieve an acceptable
                 result at 2385 cm-1, the 'bandhead' of v3.
                 Clough et al., presentation at EGU 2007
       *** mt_ckd_2.5  Adjustment to the original scaling made.
                       (temperature dependence of continuum also added)
    
    """
    bfco2 = pd.read_csv("bfco2.csv")
    bf_co2_nu = bfco2["x"]
    S_co2 = bfco2["f"]
    
    
    V1S = bf_co2_nu.min()
    V2S = bf_co2_nu.max()
    DVS = bf_co2_nu[1] - bf_co2_nu[0]
    NPTS = len(S_co2)
    
    # Correction factors for CO2 from 2000–3000 cm⁻¹ (mt_ckd_2.5),
    # stored every 2 cm⁻¹ as a 500‐element NumPy array.
    XFACCO2 = np.array([
        1.0000, 0.9998, 0.9997, 0.9996, 0.9995, 0.9994, 0.9992, 0.9991,
        0.9991, 0.9990, 0.9990, 0.9989, 0.9988, 0.9988, 0.9987, 0.9986,
        0.9985, 0.9984, 0.9983, 0.9982, 0.9981, 0.9980, 0.9979, 0.9978,
        0.9976, 0.9975, 0.9973, 0.9972, 0.9970, 0.9969, 0.9967, 0.9965,
        0.9963, 0.9961, 0.9958, 0.9956, 0.9954, 0.9951, 0.9948, 0.9946,
        0.9943, 0.9940, 0.9936, 0.9933, 0.9929, 0.9926, 0.9922, 0.9918,
        0.9913, 0.9909, 0.9904, 0.9899, 0.9894, 0.9889, 0.9884, 0.9878,
        0.9872, 0.9866, 0.9859, 0.9853, 0.9846, 0.9838, 0.9831, 0.9823,
        0.9815, 0.9806, 0.9798, 0.9789, 0.9779, 0.9770, 0.9759, 0.9749,
        0.9738, 0.9727, 0.9716, 0.9704, 0.9691, 0.9679, 0.9666, 0.9652,
        0.9638, 0.9624, 0.9609, 0.9594, 0.9578, 0.9562, 0.9546, 0.9529,
        0.9511, 0.9493, 0.9475, 0.9456, 0.9436, 0.9417, 0.9396, 0.9375,
        0.9354, 0.9332, 0.9310, 0.9287, 0.9264, 0.9240, 0.9216, 0.9191,
        0.9166, 0.9140, 0.9114, 0.9087, 0.9060, 0.9032, 0.9004, 0.8976,
        0.8947, 0.8917, 0.8887, 0.8857, 0.8827, 0.8796, 0.8764, 0.8732,
        0.8700, 0.8668, 0.8635, 0.8602, 0.8568, 0.8534, 0.8500, 0.8466,
        0.8432, 0.8397, 0.8362, 0.8327, 0.8292, 0.8257, 0.8221, 0.8186,
        0.8151, 0.8115, 0.8080, 0.8044, 0.8009, 0.7973, 0.7938, 0.7903,
        0.7868, 0.7833, 0.7799, 0.7764, 0.7730, 0.7697, 0.7663, 0.7630,
        0.7597, 0.7565, 0.7533, 0.7502, 0.7471, 0.7441, 0.7411, 0.7382,
        0.7354, 0.7326, 0.7298, 0.7272, 0.7246, 0.7221, 0.7197, 0.7173,
        0.7150, 0.7129, 0.7108, 0.7088, 0.7068, 0.7050, 0.7033, 0.7016,
        0.7001, 0.6986, 0.6973, 0.6961, 0.6949, 0.6939, 0.6930, 0.6921,
        0.6914, 0.6908, 0.6903, 0.6899, 0.6897, 0.6895, 0.6895, 0.6895,
        0.6895, 0.6895, 0.6895, 0.6908, 0.7014, 0.7121, 0.7227, 0.7552,
        0.8071, 0.8400, 0.9012, 0.9542, 1.0044, 1.0330, 1.0554, 1.0766,
        1.0967, 1.1160, 1.1346, 1.1525, 1.1700, 1.1869, 1.2035, 1.2196,
        1.2354, 1.2509, 1.2662, 1.2811, 1.2958, 1.3103, 1.3245, 1.3386,
        1.3525, 1.3661, 1.3796, 1.3930, 1.4062, 1.4193, 1.4322, 1.4449,
        1.4576, 1.4701, 1.4825, 1.4949, 1.5070, 1.5191, 1.5311, 1.5430,
        1.5548, 1.5550, 1.5550, 1.5550, 1.5550, 1.5550, 1.5550, 1.5550,
        1.5550, 1.5550, 1.5550, 1.5550, 1.5550, 1.5550, 1.5550, 1.5550,
        1.5550, 1.5550, 1.5550, 1.5550, 1.5550, 1.5550, 1.5549, 1.5547,
        1.5543, 1.5539, 1.5532, 1.5525, 1.5516, 1.5506, 1.5494, 1.5481,
        1.5467, 1.5452, 1.5435, 1.5417, 1.5397, 1.5377, 1.5355, 1.5332,
        1.5308, 1.5282, 1.5255, 1.5228, 1.5199, 1.5169, 1.5137, 1.5105,
        1.5072, 1.5037, 1.5002, 1.4966, 1.4929, 1.4890, 1.4851, 1.4811,
        1.4771, 1.4729, 1.4686, 1.4643, 1.4599, 1.4555, 1.4509, 1.4463,
        1.4417, 1.4370, 1.4322, 1.4274, 1.4225, 1.4176, 1.4126, 1.4076,
        1.4025, 1.3974, 1.3923, 1.3872, 1.3820, 1.3768, 1.3716, 1.3663,
        1.3611, 1.3558, 1.3505, 1.3452, 1.3400, 1.3347, 1.3294, 1.3241,
        1.3188, 1.3135, 1.3083, 1.3030, 1.2978, 1.2926, 1.2874, 1.2822,
        1.2771, 1.2720, 1.2669, 1.2618, 1.2568, 1.2518, 1.2468, 1.2419,
        1.2370, 1.2322, 1.2274, 1.2227, 1.2180, 1.2133, 1.2087, 1.2041,
        1.1996, 1.1952, 1.1907, 1.1864, 1.1821, 1.1778, 1.1737, 1.1695,
        1.1654, 1.1614, 1.1575, 1.1536, 1.1497, 1.1460, 1.1422, 1.1386,
        1.1350, 1.1314, 1.1280, 1.1246, 1.1212, 1.1179, 1.1147, 1.1115,
        1.1084, 1.1053, 1.1024, 1.0994, 1.0966, 1.0938, 1.0910, 1.0883,
        1.0857, 1.0831, 1.0806, 1.0781, 1.0757, 1.0734, 1.0711, 1.0688,
        1.0667, 1.0645, 1.0624, 1.0604, 1.0584, 1.0565, 1.0546, 1.0528,
        1.0510, 1.0493, 1.0476, 1.0460, 1.0444, 1.0429, 1.0414, 1.0399,
        1.0385, 1.0371, 1.0358, 1.0345, 1.0332, 1.0320, 1.0308, 1.0296,
        1.0285, 1.0275, 1.0264, 1.0254, 1.0244, 1.0235, 1.0226, 1.0217,
        1.0208, 1.0200, 1.0192, 1.0184, 1.0177, 1.0170, 1.0163, 1.0156,
        1.0150, 1.0143, 1.0137, 1.0132, 1.0126, 1.0121, 1.0116, 1.0111,
        1.0106, 1.0101, 1.0097, 1.0092, 1.0088, 1.0084, 1.0081, 1.0077,
        1.0074, 1.0070, 1.0067, 1.0064, 1.0061, 1.0058, 1.0055, 1.0053,
        1.0050, 1.0048, 1.0046, 1.0043, 1.0041, 1.0039, 1.0037, 1.0036,
        1.0034, 1.0032, 1.0030, 1.0029, 1.0027, 1.0026, 1.0025, 1.0023,
        1.0022, 1.0021, 1.0020, 1.0019, 1.0018, 1.0017, 1.0016, 1.0015,
        1.0014, 1.0014, 1.0013, 1.0012, 1.0011, 1.0011, 1.0010, 1.0010,
        1.0009, 1.0009, 1.0008, 1.0007, 1.0006, 1.0005, 1.0004, 1.0003,
        1.0002, 1.0001, 1.0000, 1.0000, 1.0000
    ])

    
    
    tdep_bandhead = np.array([
        1.44e-01, 3.61e-01, 5.71e-01, 7.63e-01, 8.95e-01,
        9.33e-01, 8.75e-01, 7.30e-01, 5.47e-01, 3.79e-01,
        2.55e-01, 1.78e-01, 1.34e-01, 1.07e-01, 9.06e-02,
        7.83e-02, 6.83e-02, 6.00e-02, 5.30e-02, 4.72e-02,
        4.24e-02, 3.83e-02, 3.50e-02, 3.23e-02, 3.01e-02
    ])
    t_eff = 246.0

    trat = tave / t_eff
    V1C_init = V1ABS - DVS
    V2C_init = V2ABS + DVS

    if V1C_init < V1S:
        I1 = -1
    else:
        I1 = int((V1C_init - V1S) / DVS + 0.01)
    V1C = V1S + DVS * (I1 - 1)
    I2 = int((V2C_init - V1S) / DVS + 0.01)

    NPTC = I2 - I1 + 3
    if NPTC > NPTS:
        NPTC = NPTS + 4
    V2C = V1C + DVS * (NPTC - 1)

    fco2 = np.zeros(NPTC, dtype=float)
    I_indices = I1 + np.arange(NPTC)

    valid_mask = np.logical_and(I_indices >= 1, I_indices <= NPTS)
    if np.any(valid_mask):
        tcor = np.ones(NPTC, dtype=float)
        band_mask = np.logical_and(I_indices >= 1196, I_indices <= 1220)# valid_mask
        if np.any(band_mask):
            idx_band = I_indices[band_mask] - 1196
            tcor[band_mask] = trat ** tdep_bandhead[idx_band]

        S_vals = np.zeros(NPTC, dtype=float)
        S_vals[valid_mask] = S_co2[I_indices[valid_mask] - 1]
        fco2 = tcor * S_vals

    VJ = V1C + DVS * np.arange(NPTC)
    CFAC = np.ones(NPTC, dtype=float)
    mask = np.logical_and(VJ >= 2000.0, VJ <= 2998.0)
    if np.any(mask):
        JFAC = (VJ[mask] - 1998.0) / 2.0 + 1e-5
        idx_axis = np.arange(1, len(XFACCO2) + 1, dtype=float)
        CFAC[mask] = np.interp(JFAC, idx_axis, XFACCO2)
    fco2 *= CFAC

    WKCO2 = air * xco2 # in molecules/cm³
    WKCO2_col = WKCO2 * (thickness * 1000 * 100) # in molecules/cm²
    WCO2 = WKCO2_col * RHOAVE * WK_CONST * XCO2C 
    C_co2 = fco2 * WCO2

    if JRAD == 1:
        C_co2 *= radfn_cal(VJ, XKT)

    return C_co2, V1C, V2C, DVS, NPTC

def frnco2_profile(abs_v1, abs_v2, abs_dv,
                   p_arr, T_arr, thickness_arr,
                   air_arr, xCO2_arr, 
                   RHOAVE_arr, XKT_arr,
                   JRAD=False):
    """
    Compute CO₂ continuum for an entire vertical profile of layers.
    """
    nlev = p_arr.size
    npt_abs = int((abs_v2 - abs_v1) / abs_dv) + 1
    C_co2_all = np.zeros((nlev, npt_abs), dtype=float)

    V1S_co2 = -4.0
    V2S_co2 = 10000.0


    for i in range(nlev):
        if abs_v1 > V2S_co2 or abs_v2 < V1S_co2:
            abs_array = np.zeros(npt_abs, dtype=float)
            
        else:
            C_co2, V1C, V2C, DVC, NPTC = frnco2_scalar(
                V1ABS=abs_v1, V2ABS=abs_v2,
                tave=T_arr[i], air=air_arr[i], thickness=thickness_arr[i],
                xco2=xCO2_arr[i],
                RHOAVE=RHOAVE_arr[i],
                XKT=XKT_arr[i],
                JRAD=JRAD
            )

            i_lo, i_hi, frac = pre_xint(
                cont_v1=V1C, cont_v2=V2C, cont_dv=DVC, npt_cont=NPTC,
                abs_v1=abs_v1, abs_v2=abs_v2, abs_dv=abs_dv, npt_abs=npt_abs
            )
            abs_array = np.zeros(npt_abs, dtype=float)
            xint(cont_values=C_co2, i_lo=i_lo, i_hi=i_hi, frac=frac, abs_array=abs_array)

        C_co2_all[i, :] = abs_array

    return C_co2_all

def xo3chp(v1abs, v2abs):
    """
    Python version of the Fortran SUBROUTINE XO3CHP.

    Parameters
    ----------
    v1abs, v2abs : float
        The ABSORB wavenumber lower/upper bounds from /ABSORB/.
    dvs : float
        Spacing (DV) from /O3CHAP/.
    x, y, z : array_like of length npts
        Tables X, Y, Z from /O3CHAP/.
    v1s, v2s : float
        V1S, V2S from /O3CHAP/.
    npts : int
        NPTS from /O3CHAP/.

    Returns
    -------
    v1c, v2c, dvc : float
        The new centered bounds and spacing.
    nptc : int
        The number of points.
    c0, c1, c2 : np.ndarray of shape (nptc,)
        The interpolated coefficient arrays.
    v1ss, v2ss : float
        Copies of the original V1S, V2S (for “ss” storage).
    """
    # set up
    bo3ch_nu_list = pd.read_csv("bo3ch.csv")["nu"].values
    bo3ch_x = pd.read_csv("bo3ch.csv")["x"].values
    bo3ch_y = pd.read_csv("bo3ch.csv")["y"].values
    bo3ch_z = pd.read_csv("bo3ch.csv")["z"].values
    v1s = bo3ch_nu_list.min()
    v2s = bo3ch_nu_list.max()
    dvs = bo3ch_nu_list[1] - bo3ch_nu_list[0]
    dvc = dvs
    npts = len(bo3ch_nu_list)

    # initial center bounds
    v1c = v1abs - dvc
    v2c = v2abs + dvc

    # compute I1
    if v1c < v1s:
        i1 = -1
    else:
        i1 = int((v1c - v1s) / dvs + 0.01)
    # re-center v1c
    v1c = v1s + dvs * (i1 - 1)

    # compute I2 and NPTC
    i2 = int((v2c - v1s) / dvs + 0.01)
    nptc = i2 - i1 + 3
    if nptc > npts:
        nptc = npts + 4
    # re-center v2c
    v2c = v1c + dvs * (nptc - 1)

    # allocate output arrays
    c0 = np.zeros(nptc, dtype=float)
    c1 = np.zeros(nptc, dtype=float)
    c2 = np.zeros(nptc, dtype=float)

    # fill
    for j in range(nptc):
        i = i1 + j
        if i < 1 or i > npts:
            # outside data range
            c0[j] = 0.0
            c1[j] = 0.0
            c2[j] = 0.0
        else:
            # remove radiation field, divide by vj
            vj = v1c + dvc * j
            c0[j] = bo3ch_x[i-1] / vj
            c1[j] = bo3ch_y[i-1] / vj
            c2[j] = bo3ch_z[i-1] / vj

    return v1c, v2c, dvc, nptc, c0, c1, c2

def o3hht0(v1abs, v2abs):
    """
    Python port of the Fortran SUBROUTINE O3HHT0.

    Arguments:
      v1abs, v2abs, dvabs, nptabs  -- from /ABSORB/ common block
      V1S, V2S, DVS, NPTS, S        -- from /O3HH0/ common block (S length >= NPTS)
      C                             -- preallocated output array (size >= NPTS+4)

    Returns:
      v1c, v2c, dvc, nptc, v1ss, v2ss
    """
    bo3hh0_data = pd.read_csv("bo3hh0.csv")
    
    # replicate Fortran: DVC = DVS; v1ss=v1s; v2ss=v2s
    V1S = bo3hh0_data["x"].min()
    V2S = bo3hh0_data["x"].max()
    DVS = bo3hh0_data["x"][1] - bo3hh0_data["x"][0]
    NPTS = len(bo3hh0_data["x"])
    S = bo3hh0_data["f"].values
    dvc = DVS
    v1ss = V1S
    v2ss = V2S

    # initial coarse grid
    v1c = v1abs - dvc
    v2c = v2abs + dvc

    # compute I1
    if v1c < V1S:
        i1 = -1
    else:
        i1 = int((v1c - V1S) / DVS + 0.01)

    # snap v1c to the grid
    v1c = V1S + DVS * (i1 - 1)

    # compute I2 and number of points
    i2 = int((v2c - V1S) / DVS + 0.01)
    nptc = i2 - i1 + 3
    if nptc > NPTS:
        nptc = NPTS + 4
        
    C = np.zeros(nptc, dtype=float)

    # recompute upper bound
    v2c = v1c + DVS * (nptc - 1)

    # fill C[0..nptc-1]
    for j in range(nptc):
        idx = i1 + j
        if idx < 1 or idx > NPTS:
            C[j] = 0.0
        else:
            vj = v1c + dvc * j
            # Fortran S(I) is S[idx-1] in Python
            C[j] = S[idx-1] / vj

    return v1c, v2c, dvc, nptc, C

def o3hht1(v1abs, v2abs):
    """
    Python port of Fortran SUBROUTINE O3HHT1.
    Inputs from common blocks:
      v1abs, v2abs,      -- V1ABS, V2ABS
      DVS, V1S, NPTS, S  -- O3HH1 block
    Output in-place array C, plus (v1c, v2c, dvc, nptc).
    """
    # set up
    bo3hh1_data = pd.read_csv("bo3hh1.csv")
    
    # replicate Fortran: DVC = DVS; v1ss=v1s; v2ss=v2s
    V1S = bo3hh1_data["x"].min()
    V2S = bo3hh1_data["x"].max()
    DVS = bo3hh1_data["x"][1] - bo3hh1_data["x"][0]
    NPTS = len(bo3hh1_data["x"])
    S = bo3hh1_data["f"].values
    
    dvc = DVS
    v1c = v1abs - dvc
    v2c = v2abs + dvc

    # find starting index I1
    if v1c < V1S:
        i1 = -1
    else:
        i1 = int((v1c - V1S) / DVS + 0.01)

    # snap v1c to grid
    v1c = V1S + DVS * (i1 - 1)

    # compute number of points
    i2 = int((v2c - V1S) / DVS + 0.01)
    nptc = i2 - i1 + 3
    if nptc > NPTS:
        nptc = NPTS + 4
        
    C = np.zeros(nptc, dtype=float)

    # recompute v2c
    v2c = v1c + DVS * (nptc - 1)

    # fill C[0..nptc-1]
    for j in range(nptc):
        idx = i1 + j
        if 1 <= idx <= NPTS:
            C[j] = S[idx-1]
        else:
            C[j] = 0.0

    return v1c, v2c, dvc, nptc, C


def o3hht2(v1abs, v2abs):
    """
    Python port of Fortran SUBROUTINE O3HHT2.
    Same structure as O3HHT1 but uses the O3HH2 data block.
    """
    # set up
    bo3hh2_data = pd.read_csv("bo3hh2.csv")
    
    # replicate Fortran: DVC = DVS; v1ss=v1s; v2ss=v2s
    V1S = bo3hh2_data["x"].min()
    V2S = bo3hh2_data["x"].max()
    DVS = bo3hh2_data["x"][1] - bo3hh2_data["x"][0]
    NPTS = len(bo3hh2_data["x"])
    S = bo3hh2_data["f"].values
    dvc = DVS
    v1c = v1abs - dvc
    v2c = v2abs + dvc

    # find starting index I1
    if v1c < V1S:
        i1 = -1
    else:
        i1 = int((v1c - V1S) / DVS + 0.01)

    # snap v1c to grid
    v1c = V1S + DVS * (i1 - 1)

    # compute number of points
    i2 = int((v2c - V1S) / DVS + 0.01)
    nptc = i2 - i1 + 3
    if nptc > NPTS:
        nptc = NPTS + 4

    # recompute v2c
    v2c = v1c + DVS * (nptc - 1)
    
    C = np.zeros(nptc, dtype=float)

    # fill C[0..nptc-1]
    for j in range(nptc):
        idx = i1 + j
        if 1 <= idx <= NPTS:
            C[j] = S[idx-1]
        else:
            C[j] = 0.0

    return v1c, v2c, dvc, nptc, C

def o3hhuv(v1abs, v2abs):
    """
    Python port of the Fortran SUBROUTINE O3HHUV.

    Parameters
    ----------
    v1abs, v2abs : float
        V1ABS, V2ABS from the /ABSORB/ common block.


    Returns
    -------
    v1c, v2c, dvc, nptc, v1ss, v2ss : tuple
        - v1c, v2c : float
            adjusted grid endpoints,
        - dvc : float
            grid spacing (== DVS),
        - nptc : int
            number of output points,
        - v1ss, v2ss : float
            copies of V1S, V2S for interpolation bookkeeping.
    """
    # set up
    
    bo3huv_nu = np.arange(40800., 54000.+1e-6, 100.)

    bo3huv_f = np.array([
                9.91204E-18, 9.76325E-18, 9.72050E-18, 9.51049E-18, 9.23530E-18,
                9.02306E-18, 8.90510E-18, 8.60115E-18, 8.39094E-18, 8.27926E-18,
                7.95525E-18, 7.73583E-18, 7.55018E-18, 7.31076E-18, 7.10415E-18,
                6.87747E-18, 6.66639E-18, 6.39484E-18, 6.27101E-18, 6.01019E-18,
                5.77594E-18, 5.60403E-18, 5.40837E-18, 5.21289E-18, 4.99329E-18,
                4.81742E-18, 4.61608E-18, 4.45707E-18, 4.28261E-18, 4.09672E-18,
                3.93701E-18, 3.77835E-18, 3.61440E-18, 3.45194E-18, 3.30219E-18,
                3.15347E-18, 3.01164E-18, 2.87788E-18, 2.74224E-18, 2.61339E-18,
                2.48868E-18, 2.36872E-18, 2.25747E-18, 2.14782E-18, 2.03997E-18,
                1.94281E-18, 1.84525E-18, 1.75275E-18, 1.67151E-18, 1.58813E-18,
                1.50725E-18, 1.43019E-18, 1.35825E-18, 1.28878E-18, 1.22084E-18,
                1.15515E-18, 1.09465E-18, 1.03841E-18, 9.83780E-19, 9.31932E-19,
                8.83466E-19, 8.38631E-19, 7.96631E-19, 7.54331E-19, 7.13805E-19,
                6.78474E-19, 6.44340E-19, 6.13104E-19, 5.81777E-19, 5.53766E-19,
                5.27036E-19, 5.03555E-19, 4.82633E-19, 4.61483E-19, 4.42014E-19,
                4.23517E-19, 4.07774E-19, 3.93060E-19, 3.80135E-19, 3.66348E-19,
                3.53665E-19, 3.47884E-19, 3.39690E-19, 3.34288E-19, 3.29135E-19,
                3.23104E-19, 3.18875E-19, 3.16800E-19, 3.15925E-19, 3.12932E-19,
                3.12956E-19, 3.15522E-19, 3.14950E-19, 3.15924E-19, 3.19059E-19,
                3.23109E-19, 3.27873E-19, 3.33788E-19, 3.39804E-19, 3.44925E-19,
                3.50502E-19, 3.55853E-19, 3.59416E-19, 3.68933E-19, 3.78284E-19,
                3.86413E-19, 3.98049E-19, 4.04700E-19, 4.12958E-19, 4.23482E-19,
                4.31203E-19, 4.41885E-19, 4.52651E-19, 4.61492E-19, 4.70493E-19,
                4.80497E-19, 4.90242E-19, 4.99652E-19, 5.10316E-19, 5.21510E-19,
                5.32130E-19, 5.43073E-19, 5.56207E-19, 5.61756E-19, 5.66799E-19,
                5.85545E-19, 5.92409E-19, 5.96168E-19, 6.12497E-19, 6.20231E-19,
                6.24621E-19, 6.34160E-19, 6.43622E-19,
                ])
    
    # replicate Fortran: DVC = DVS; v1ss=v1s; v2ss=v2s
    V1S = bo3huv_nu.min()
    V2S = bo3huv_nu.max()
    DVS = bo3huv_nu[1] - bo3huv_nu[0]
    NPTS = len(bo3huv_nu)
    S = bo3huv_f
    
    dvc = DVS
    v1ss = V1S
    v2ss = V2S
    v1c = v1abs - dvc
    v2c = v2abs + dvc

    # compute starting index I1
    if v1c < V1S:
        i1 = -1
    else:
        i1 = int((v1c - V1S) / DVS + 0.01)

    # snap v1c to the exact grid
    v1c = V1S + DVS * (i1 - 1)

    # compute ending index and nptc
    i2 = int((v2c - V1S) / DVS + 0.01)
    nptc = i2 - i1 + 3
    if nptc > NPTS:
        nptc = NPTS + 4
    
    C = np.zeros(nptc, dtype=float)

    # recompute v2c
    v2c = v1c + DVS * (nptc - 1)

    # fill C[0..nptc-1]
    for j in range(nptc):
        idx = i1 + j
        if 1 <= idx <= NPTS:
            vj = v1c + dvc * j
            C[j] = S[idx - 1] / vj
        else:
            C[j] = 0.0

    return v1c, v2c, dvc, nptc, C

def o3_diffuse(V1ABS, V2ABS, xo3cn,
                tave, thickness, air, xo3, XKT, RHOAVE,
                JRAD
                ):
    """
    Compute the diffuse ozone (XO3CHP) contribution.
    """
    
    # 1) Chapman‐diffuse smoothing over 9170–24565 cm⁻¹
    if V2ABS > 8920.0 and V1ABS <= 24665.0 and xo3cn > 0.0:

        WKO3 = air * xo3 # in molecules/cm³
        WKO3_col = WKO3 * (thickness * 1000 * 100) # in molecules/cm²
        WO3 = WKO3_col * WK_CONST * xo3cn * RHOAVE
        V1C, V2C, DVC, NPTO3, cch0, cch1, cch2 = xo3chp(V1ABS, V2ABS)

        DT = tave - 273.15
        for j in range(NPTO3):
            cch0[j] = (cch0[j] + (cch1[j] + cch2[j] * DT) * DT) * WO3
            VJ = V1C + DVC * j
            if JRAD == 1:
                cch0[j] *= radfn_cal(VJ, XKT)
        
        c_final = cch0
    
    # 2) Mid‐UV (27370–40800 cm⁻¹) Hartley–Huggins
    elif V2ABS > 27370.0 and V1ABS < 40800.0 and xo3cn > 0.0:

        WKO3 = air * xo3 # in molecules/cm³
        WKO3_col = WKO3 * (thickness * 1000 * 100) # in molecules/cm²
        WO3 = WKO3_col * WK_CONST * xo3cn
        TC = tave - 273.15

        V1C, V2C, DVC, NPTO3, C0 = o3hht0(V1ABS, V2ABS)
        # fill CT1, CT2 over their own grids
        V1C, V2C, DVC, NPTO3, CT1 = o3hht1(V1ABS, V2ABS)   # pass your V1T1,V2T1,DVT1,NPT1
        V1C, V2C, DVC, NPTO3, CT2 = o3hht2(V1ABS, V2ABS)   # pass your V1T2,V2T2,DVT2,NPT2

        for j in range(NPTO3):
            val = C0[j] * WO3
            VJ = V1C + DVC * j
            if JRAD == 1:
                val *= radfn_cal(VJ, XKT)
            C[j] = val * (1.0 + CT1[j] * TC + CT2[j] * TC ** 2)

        c_final = C

    # 3) UV Hartley (>40800 cm⁻¹ up to 54 000)
    elif V2ABS > 40800.0 and V1ABS < 54000.0 and xo3cn > 0.0:
        C0 = np.zeros(NPTO3)
        C  = np.zeros(NPTO3)

        WKO3 = air * xo3 # in molecules/cm³
        WKO3_col = WKO3 * (thickness * 1000 * 100) # in molecules/cm²
        WO3 = WKO3_col * xo3cn
        V1C, V2C, DVC, NPTO3, C = o3hhuv(V1ABS, V2ABS)

        for j in range(NPTO3):
            val = C0[j] * WO3
            VJ = V1C + DVC * j
            if JRAD == 1:
                val *= radfn_cal(VJ, XKT)
            C[j] = val
        
        c_final = C
    
    else:
        V1C = None
        V2C = None
        DVC = None
        DVC = None
        NPTO3 = None
        c_final = None

    return V1C, V2C, DVC, NPTO3, c_final

def o3_diffuse_profile(abs_v1, abs_v2, abs_dv,
                   p_arr, T_arr, thickness_arr,
                   air_arr, xO3_arr, 
                   RHOAVE_arr, XKT_arr,
                   JRAD=False):
    """
    Compute diffuse ozone for an entire vertical profile of layers.
    """
    nlev = p_arr.size
    npt_abs = int((abs_v2 - abs_v1) / abs_dv) + 1
    C_o3_diff_all = np.zeros((nlev, npt_abs), dtype=float)


    for i in range(nlev):
        V1C, V2C, DVC, NPTO3, C_O3_diff = o3_diffuse(
            V1ABS=abs_v1, V2ABS=abs_v2, xo3cn=XO3CN,
            tave=T_arr[i], thickness=thickness_arr[i],
            air=air_arr[i], xo3=xO3_arr[i],
            XKT=XKT_arr[i], RHOAVE=RHOAVE_arr[i],
            JRAD=JRAD
                    )
        if C_O3_diff is not None:
            i_lo, i_hi, frac = pre_xint(
                cont_v1=V1C, cont_v2=V2C, cont_dv=DVC, npt_cont=NPTO3,
                abs_v1=abs_v1, abs_v2=abs_v2, abs_dv=abs_dv, npt_abs=npt_abs
            )
            abs_array = np.zeros(npt_abs, dtype=float)
            xint(cont_values=C_O3_diff, i_lo=i_lo, i_hi=i_hi, frac=frac, abs_array=abs_array)

        else:
            abs_array = np.zeros(npt_abs, dtype=float)
            
            
        C_o3_diff_all[i, :] = abs_array

    return C_o3_diff_all, V1C, V2C, DVC, NPTO3

def o2_ver_1(v1abs, v2abs, T):
    """
    Port of the Fortran SUBROUTINE o2_ver_1.

    Returns (v1c, v2c, dvc, nptc, C, v1ss, v2ss) where C is an np.ndarray of length nptc.
    
            

            The temperature correction is done in this function

            c0 are the oxygen absorption coefficients at temperature
            tave
                - these absorption coefficients are in units of
                    [(cm^2/molec) 10^20)]/(cm-1  amagat)
                - cm-1 in the denominator arises through the removal
                    of the radiation field
                - for this case, an amagat is interpreted as one
                    loschmidt of air (273K)
    
    
    """
    
    bo2f_nu = np.arange(1340.000, 1850.000+1e-5, 5.000)

    bo2f_o0 = np.array([
        0.000E+00,  9.744E-09,  2.256E-08,  3.538E-08,  4.820E-08,
        6.100E-08,  7.400E-08,  8.400E-08,  9.600E-08,  1.200E-07,
        1.620E-07,  2.080E-07,  2.460E-07,  2.850E-07,  3.140E-07,
        3.800E-07,  4.440E-07,  5.000E-07,  5.710E-07,  6.730E-07,
        7.680E-07,  8.530E-07,  9.660E-07,  1.100E-06,  1.210E-06,
        1.330E-06,  1.470E-06,  1.590E-06,  1.690E-06,  1.800E-06,
        1.920E-06,  2.040E-06,  2.150E-06,  2.260E-06,  2.370E-06,
        2.510E-06,  2.670E-06,  2.850E-06,  3.070E-06,  3.420E-06,
        3.830E-06,  4.200E-06,  4.450E-06,  4.600E-06,  4.530E-06,
        4.280E-06,  3.960E-06,  3.680E-06,  3.480E-06,  3.350E-06,
        3.290E-06,  3.250E-06,  3.230E-06,  3.230E-06,  3.210E-06,
        3.190E-06,  3.110E-06,  3.030E-06,  2.910E-06,  2.800E-06,
        2.650E-06,  2.510E-06,  2.320E-06,  2.130E-06,  1.930E-06,
        1.760E-06,  1.590E-06,  1.420E-06,  1.250E-06,  1.110E-06,
        9.900E-07,  8.880E-07,  7.910E-07,  6.780E-07,  5.870E-07,
        5.240E-07,  4.640E-07,  4.030E-07,  3.570E-07,  3.200E-07,
        2.900E-07,  2.670E-07,  2.420E-07,  2.150E-07,  1.820E-07,
        1.600E-07,  1.460E-07,  1.280E-07,  1.030E-07,  8.700E-08,
        8.100E-08,  7.100E-08,  6.400E-08,  5.807E-08,  5.139E-08,
        4.496E-08,  3.854E-08,  3.212E-08,  2.569E-08,  1.927E-08,
        1.285E-08,  6.423E-09,  0.000E+00,
    ])

    bo2f_ot = np.array([
        4.000E+02,  4.000E+02,  4.000E+02,  4.000E+02,  4.000E+02,
        4.670E+02,  4.000E+02,  3.150E+02,  3.790E+02,  3.680E+02,
        4.750E+02,  5.210E+02,  5.310E+02,  5.120E+02,  4.420E+02,
        4.440E+02,  4.300E+02,  3.810E+02,  3.350E+02,  3.240E+02,
        2.960E+02,  2.480E+02,  2.150E+02,  1.930E+02,  1.580E+02,
        1.270E+02,  1.010E+02,  7.100E+01,  3.100E+01, -6.000E+00,
        -2.600E+01, -4.700E+01, -6.300E+01, -7.900E+01, -8.800E+01,
        -8.800E+01, -8.700E+01, -9.000E+01, -9.800E+01, -9.900E+01,
        -1.090E+02, -1.340E+02, -1.600E+02, -1.670E+02, -1.640E+02,
        -1.580E+02, -1.530E+02, -1.510E+02, -1.560E+02, -1.660E+02,
        -1.680E+02, -1.730E+02, -1.700E+02, -1.610E+02, -1.450E+02,
        -1.260E+02, -1.080E+02, -8.400E+01, -5.900E+01, -2.900E+01,
        4.000E+00,  4.100E+01,  7.300E+01,  9.700E+01,  1.230E+02,
        1.590E+02,  1.980E+02,  2.200E+02,  2.420E+02,  2.560E+02,
        2.810E+02,  3.110E+02,  3.340E+02,  3.190E+02,  3.130E+02,
        3.210E+02,  3.230E+02,  3.100E+02,  3.150E+02,  3.200E+02,
        3.350E+02,  3.610E+02,  3.780E+02,  3.730E+02,  3.380E+02,
        3.190E+02,  3.460E+02,  3.220E+02,  2.910E+02,  2.900E+02,
        3.500E+02,  3.710E+02,  5.040E+02,  4.000E+02,  4.000E+02,
        4.000E+02,  4.000E+02,  4.000E+02,  4.000E+02,  4.000E+02,
        4.000E+02,  4.000E+02,  4.000E+02,
    ])
    
    V1S = bo2f_nu.min()
    V2S = bo2f_nu.max()
    DVS = bo2f_nu[1] - bo2f_nu[0]
    NPTS = len(bo2f_nu)
    xo2 = bo2f_o0
    xo2t = bo2f_ot
    
    # physical constants from the DATA statement
    T0     = 296.0
    XLOSMT = 2.68675e19    # (molecules/cm² at 1 amagat)
    # temperature factor
    xktfac = (1.0 / T0) - (1.0 / T)
    # LBLRTM‐consistent prefactor
    factor = 1.0e20 / XLOSMT

    # initialize grid endpoints
    dvc  = DVS

    v1c  = v1abs - dvc
    v2c  = v2abs + dvc

    # figure out integer start index I1
    if v1c < V1S:
        i1 = -1
    else:
        i1 = int((v1c - V1S)/DVS + 0.01)

    # snap v1c back onto the exact grid
    v1c = V1S + DVS * (i1 - 1)
    # end index
    i2 = int((v2c - V1S)/DVS + 0.01)
    # number of pts plus 2‐point “padding”
    nptc = i2 - i1 + 3
    if nptc > NPTS:
        nptc = NPTS + 4
    # recompute v2c
    v2c = v1c + DVS * (nptc - 1)

    # allocate and fill C
    C = np.zeros(nptc, dtype=float)
    for j in range(nptc):
        idx = i1 + j
        if 1 <= idx <= NPTS:
            vj = v1c + dvc * j
            C[j] = factor * xo2[idx-1] * np.exp(xo2t[idx-1]*xktfac) / vj

    return v1c, v2c, dvc, nptc, C


def o2_ver1_profile(abs_v1, abs_v2, abs_dv,
                   p_arr, T_arr, thickness_arr,
                   air_arr, xO2_arr,
                   RHOAVE_arr, XKT_arr, amagat_arr,
                   JRAD=False
                    ):
    """
    version_1 of the Oxygen Collision Induced Fundamental
    
    F. Thibault, V. Menoux, R. Le Doucen, L. Rosenman, J.-M. Hartmann, and Ch. Boulet,
    Infrared collision-induced absorption by O2 near 6.4 microns
    for atmospheric applications: measurements and emprirical
    modeling, Appl. Optics, 35, 5911-5917, (1996).

    Mirror of the Fortran “if (V2>1340 .and. V1<1850 .and. xo2cn>0) then …” block.
    Returns (CDF, v1c, dvc, nptc) where CDF is the final absorption array
    merged into ABSRB.
    """
    
    nlev = p_arr.size
    npt_abs = int((abs_v2 - abs_v1) / abs_dv) + 1
    C_o2_all = np.zeros((nlev, npt_abs), dtype=float)
    
    # only run in the valid window
    if not (abs_v2 > 1340.0 and abs_v1 < 1850.0 and XO2CN > 0.0):
        return C_o2_all

    
    for i in range(nlev):
        # scale factor for column amount & units
        WKO2 = air_arr[i] * xO2_arr[i] # in molec/cm³
        WKO2_col = WKO2 * (thickness_arr[i] * 1000 * 100) # in molec/cm²        
        
        tau_fac = XO2CN * WKO2_col * WK_CONST * amagat_arr[i]
        
        #   WkO2 is the oxygen column amount in units of molec/cm2
        #    amagat is in units of amagats (air)
        
        # compute the per‐point O2 absorption at TAVE
        v1c, v2c, dvc, nptc_o2_ver_1, C_O2_ver_1 = o2_ver_1(v1abs=abs_v1, v2abs=abs_v2, T=T_arr[i])
        
        # apply column scaling + radiation‐field if needed
        C = tau_fac * C_O2_ver_1
        if JRAD == 1:
            from math import exp
            for j in range(nptc_o2_ver_1):
                vj = v1c + dvc*j
                C[j] *= radfn_cal(vj, XKT_arr[i])
        
        if C_O2_ver_1 is not None:
            i_lo, i_hi, frac = pre_xint(
                cont_v1=v1c, cont_v2=v2c, cont_dv=dvc, npt_cont=nptc_o2_ver_1,
                abs_v1=abs_v1, abs_v2=abs_v2, abs_dv=abs_dv, npt_abs=npt_abs
            )
            abs_array = np.zeros(npt_abs, dtype=float)
            xint(cont_values=C_O2_ver_1, i_lo=i_lo, i_hi=i_hi, frac=frac, abs_array=abs_array)

        else:
            abs_array = np.zeros(npt_abs, dtype=float)

        C_o2_all[i, :] = abs_array

    return C_o2_all

def o2inf1(v1abs, v2abs):
    """Mate et al. continuum (7550–8486 cm⁻¹)."""
    
    bo2inf1_nu = np.arange(7536.000, 8500.000+1e-5, 2.000)

    bo2inf1 = np.array([
        0.000E+00,  4.355E-11,  8.709E-11,  1.742E-10,  3.484E-10,
        6.968E-10,  1.394E-09,  2.787E-09,  3.561E-09,  3.314E-09,
        3.368E-09,  3.435E-09,  2.855E-09,  3.244E-09,  3.447E-09,
        3.891E-09,  4.355E-09,  3.709E-09,  4.265E-09,  4.772E-09,
        4.541E-09,  4.557E-09,  4.915E-09,  4.688E-09,  5.282E-09,
        5.755E-09,  5.096E-09,  5.027E-09,  4.860E-09,  4.724E-09,
        5.048E-09,  5.248E-09,  5.473E-09,  4.852E-09,  5.362E-09,
        6.157E-09,  6.150E-09,  6.347E-09,  6.388E-09,  6.213E-09,
        6.521E-09,  8.470E-09,  8.236E-09,  8.269E-09,  8.776E-09,
        9.122E-09,  9.189E-09,  9.778E-09,  8.433E-09,  9.964E-09,
        9.827E-09,  1.064E-08,  1.063E-08,  1.031E-08,  1.098E-08,
        1.156E-08,  1.295E-08,  1.326E-08,  1.467E-08,  1.427E-08,
        1.452E-08,  1.456E-08,  1.554E-08,  1.605E-08,  1.659E-08,
        1.754E-08,  1.757E-08,  1.876E-08,  1.903E-08,  1.876E-08,
        1.869E-08,  2.036E-08,  2.203E-08,  2.221E-08,  2.284E-08,
        2.288E-08,  2.394E-08,  2.509E-08,  2.663E-08,  2.720E-08,
        2.839E-08,  2.923E-08,  2.893E-08,  2.949E-08,  2.962E-08,
        3.057E-08,  3.056E-08,  3.364E-08,  3.563E-08,  3.743E-08,
        3.813E-08,  3.946E-08,  4.082E-08,  4.201E-08,  4.297E-08,
        4.528E-08,  4.587E-08,  4.704E-08,  4.962E-08,  5.115E-08,
        5.341E-08,  5.365E-08,  5.557E-08,  5.891E-08,  6.084E-08,
        6.270E-08,  6.448E-08,  6.622E-08,  6.939E-08,  7.233E-08,
        7.498E-08,  7.749E-08,  8.027E-08,  8.387E-08,  8.605E-08,
        8.888E-08,  9.277E-08,  9.523E-08,  9.880E-08,  1.037E-07,
        1.076E-07,  1.114E-07,  1.151E-07,  1.203E-07,  1.246E-07,
        1.285E-07,  1.345E-07,  1.408E-07,  1.465E-07,  1.519E-07,
        1.578E-07,  1.628E-07,  1.685E-07,  1.760E-07,  1.847E-07,
        1.929E-07,  2.002E-07,  2.070E-07,  2.177E-07,  2.262E-07,
        2.365E-07,  2.482E-07,  2.587E-07,  2.655E-07,  2.789E-07,
        2.925E-07,  3.023E-07,  3.153E-07,  3.296E-07,  3.409E-07,
        3.532E-07,  3.680E-07,  3.859E-07,  3.951E-07,  4.074E-07,
        4.210E-07,  4.381E-07,  4.588E-07,  4.792E-07,  4.958E-07,
        5.104E-07,  5.271E-07,  5.501E-07,  5.674E-07,  5.913E-07,
        6.243E-07,  6.471E-07,  6.622E-07,  6.831E-07,  6.987E-07,
        7.159E-07,  7.412E-07,  7.698E-07,  7.599E-07,  7.600E-07,
        7.918E-07,  8.026E-07,  8.051E-07,  8.049E-07,  7.914E-07,
        7.968E-07,  7.945E-07,  7.861E-07,  7.864E-07,  7.741E-07,
        7.675E-07,  7.592E-07,  7.400E-07,  7.362E-07,  7.285E-07,
        7.173E-07,  6.966E-07,  6.744E-07,  6.597E-07,  6.413E-07,
        6.265E-07,  6.110E-07,  5.929E-07,  5.717E-07,  5.592E-07,
        5.411E-07,  5.235E-07,  5.061E-07,  4.845E-07,  4.732E-07,
        4.593E-07,  4.467E-07,  4.328E-07,  4.161E-07,  4.035E-07,
        3.922E-07,  3.820E-07,  3.707E-07,  3.585E-07,  3.475E-07,
        3.407E-07,  3.317E-07,  3.226E-07,  3.134E-07,  3.016E-07,
        2.969E-07,  2.894E-07,  2.814E-07,  2.749E-07,  2.657E-07,
        2.610E-07,  2.536E-07,  2.467E-07,  2.394E-07,  2.337E-07,
        2.302E-07,  2.241E-07,  2.191E-07,  2.140E-07,  2.093E-07,
        2.052E-07,  1.998E-07,  1.963E-07,  1.920E-07,  1.862E-07,
        1.834E-07,  1.795E-07,  1.745E-07,  1.723E-07,  1.686E-07,
        1.658E-07,  1.629E-07,  1.595E-07,  1.558E-07,  1.523E-07,
        1.498E-07,  1.466E-07,  1.452E-07,  1.431E-07,  1.408E-07,
        1.381E-07,  1.362E-07,  1.320E-07,  1.298E-07,  1.262E-07,
        1.247E-07,  1.234E-07,  1.221E-07,  1.197E-07,  1.176E-07,
        1.142E-07,  1.121E-07,  1.099E-07,  1.081E-07,  1.073E-07,
        1.061E-07,  1.041E-07,  1.019E-07,  9.969E-08,  9.727E-08,
        9.642E-08,  9.487E-08,  9.318E-08,  9.116E-08,  9.046E-08,
        8.827E-08,  8.689E-08,  8.433E-08,  8.324E-08,  8.204E-08,
        8.036E-08,  7.951E-08,  7.804E-08,  7.524E-08,  7.392E-08,
        7.227E-08,  7.176E-08,  6.975E-08,  6.914E-08,  6.859E-08,
        6.664E-08,  6.506E-08,  6.368E-08,  6.262E-08,  6.026E-08,
        6.002E-08,  5.866E-08,  5.867E-08,  5.641E-08,  5.589E-08,
        5.499E-08,  5.309E-08,  5.188E-08,  5.139E-08,  4.991E-08,
        4.951E-08,  4.833E-08,  4.640E-08,  4.524E-08,  4.479E-08,
        4.304E-08,  4.228E-08,  4.251E-08,  4.130E-08,  3.984E-08,
        3.894E-08,  3.815E-08,  3.732E-08,  3.664E-08,  3.512E-08,
        3.463E-08,  3.503E-08,  3.218E-08,  3.253E-08,  3.107E-08,
        2.964E-08,  2.920E-08,  2.888E-08,  2.981E-08,  2.830E-08,
        2.750E-08,  2.580E-08,  2.528E-08,  2.444E-08,  2.378E-08,
        2.413E-08,  2.234E-08,  2.316E-08,  2.199E-08,  2.088E-08,
        1.998E-08,  1.920E-08,  1.942E-08,  1.859E-08,  1.954E-08,
        1.955E-08,  1.749E-08,  1.720E-08,  1.702E-08,  1.521E-08,
        1.589E-08,  1.469E-08,  1.471E-08,  1.543E-08,  1.433E-08,
        1.298E-08,  1.274E-08,  1.226E-08,  1.204E-08,  1.201E-08,
        1.298E-08,  1.220E-08,  1.220E-08,  1.096E-08,  1.080E-08,
        9.868E-09,  9.701E-09,  1.130E-08,  9.874E-09,  9.754E-09,
        9.651E-09,  9.725E-09,  8.413E-09,  7.705E-09,  7.846E-09,
        8.037E-09,  9.163E-09,  8.098E-09,  8.160E-09,  7.511E-09,
        7.011E-09,  6.281E-09,  6.502E-09,  7.323E-09,  7.569E-09,
        5.941E-09,  5.867E-09,  5.676E-09,  4.840E-09,  5.063E-09,
        5.207E-09,  4.917E-09,  5.033E-09,  5.356E-09,  3.795E-09,
        4.983E-09,  4.600E-09,  3.635E-09,  3.099E-09,  2.502E-09,
        3.823E-09,  3.464E-09,  4.332E-09,  3.612E-09,  3.682E-09,
        3.709E-09,  3.043E-09,  3.593E-09,  3.995E-09,  4.460E-09,
        3.583E-09,  3.290E-09,  3.132E-09,  2.812E-09,  3.109E-09,
        3.874E-09,  3.802E-09,  4.024E-09,  3.901E-09,  2.370E-09,
        1.821E-09,  2.519E-09,  4.701E-09,  3.855E-09,  4.685E-09,
        5.170E-09,  4.387E-09,  4.148E-09,  4.043E-09,  3.545E-09,
        3.392E-09,  3.609E-09,  4.635E-09,  3.467E-09,  2.558E-09,
        3.389E-09,  2.672E-09,  2.468E-09,  1.989E-09,  2.816E-09,
        4.023E-09,  2.664E-09,  2.219E-09,  3.169E-09,  1.654E-09,
        3.189E-09,  2.535E-09,  2.618E-09,  3.265E-09,  2.138E-09,
        1.822E-09,  2.920E-09,  2.002E-09,  1.300E-09,  3.764E-09,
        3.212E-09,  3.222E-09,  2.961E-09,  2.108E-09,  1.708E-09,
        2.636E-09,  2.937E-09,  2.939E-09,  2.732E-09,  2.218E-09,
        1.046E-09,  6.419E-10,  1.842E-09,  1.112E-09,  1.265E-09,
        4.087E-09,  2.044E-09,  1.022E-09,  5.109E-10,  2.554E-10,
        1.277E-10,  6.386E-11,  0.000E+00])
    
    V1S = bo2inf1_nu.min()
    V2S = bo2inf1_nu.max()
    DVS = bo2inf1_nu[1] - bo2inf1_nu[0]
    NPTS = len(bo2inf1_nu)
    xo2inf1 = bo2inf1
    
    dvc = DVS
    v1c = v1abs - dvc
    v2c = v2abs + dvc

    i1 = -1 if v1c < V1S else int((v1c - V1S)/DVS + 0.01)
    v1c = V1S + DVS*(i1 - 1)
    i2 = int((v2c - V1S)/DVS + 0.01)
    nptc = i2 - i1 + 3
    if nptc > NPTS: 
        nptc = NPTS + 4
    v2c = v1c + DVS*(nptc - 1)

    C = np.zeros(nptc)
    for j in range(nptc):
        idx = i1 + j
        if 1 <= idx <= NPTS:
            vj = v1c + dvc*j
            C[j] = xo2inf1[idx-1] / vj
    freqs = v1c + dvc*np.arange(nptc)
    
    return freqs, C, v1c, v2c, dvc, nptc

def o2inf2(v1abs, v2abs):
    """Mlawer et al. continuum (9100–11000 cm⁻¹)."""
    # DATA lines
    V1_osc, HW1 = 9375.0, 58.96
    V2_osc, HW2 = 9439.0, 45.04
    S1, S2       = 1.166e-4, 3.086e-5

    V1S, V2S, DVS = 9100.0, 11000.0, 2.0
    dvc = DVS
    v1c = v1abs - dvc
    v2c = v2abs + dvc

    # clamp extremes
    if v1c < V1S: v1c = V1S - 2*DVS
    if v2c > V2S: v2c = V2S + 2*DVS

    nptc = int((v2c - v1c)/dvc + 3.01)
    v2c = v1c + dvc*(nptc - 1)

    C = np.zeros(nptc)
    for j in range(nptc):
        vj = v1c + dvc*j
        if V1S < vj < V2S:
            dv1 = vj - V1_osc
            dv2 = vj - V2_osc
            damp1 = np.exp(dv1/176.1) if dv1<0 else 1.0
            damp2 = np.exp(dv2/176.1) if dv2<0 else 1.0
            o2inf = 0.31831 * (
                (S1*damp1/HW1)/(1+(dv1/HW1)**2)
                + (S2*damp2/HW2)/(1+(dv2/HW2)**2)
            ) * 1.054
            C[j] = o2inf / vj
    freqs = v1c + dvc*np.arange(nptc)
    return freqs, C, v1c, v2c, dvc, nptc

def o2inf3(v1abs, v2abs):
    """Mlawer’s A-band continuum (12961.5–13221.5 cm⁻¹)."""
    
    bo2inf3_nu_list = np.arange(12961.5, 13221.5+1e-5, 1.0)

    bo2inf3 = np.array([
        0.000e+00,  1.253e-10,  2.785e-10,  4.316e-10,  5.848e-10,
        7.379e-10,  8.911e-10,  1.044e-09,  1.197e-09,  1.351e-09,
        1.504e-09,  1.657e-09,  1.810e-09,  1.963e-09,  2.116e-09,
        2.269e-09,  2.423e-09,  2.576e-09,  2.729e-09,  2.882e-09,
        3.035e-09,  3.188e-09,  3.342e-09,  3.495e-09,  3.648e-09,
        3.801e-09,  3.954e-09,  4.107e-09,  4.260e-09,  4.414e-09,
        4.567e-09,  4.720e-09,  4.873e-09,  5.026e-09,  5.179e-09,
        5.333e-09,  5.486e-09,  5.639e-09,  5.792e-09,  5.945e-09,
        6.098e-09,  6.251e-09,  6.405e-09,  6.552e-09,  6.738e-09,
        7.008e-09,  7.216e-09,  7.353e-09,  7.427e-09,  7.687e-09,
        7.917e-09,  8.114e-09,  8.273e-09,  8.403e-09,  8.496e-09,
        8.591e-09,  8.946e-09,  9.307e-09,  9.641e-09,  9.988e-09,
        1.033e-08,  1.065e-08,  1.105e-08,  1.156e-08,  1.221e-08,
        1.301e-08,  1.395e-08,  1.490e-08,  1.588e-08,  1.688e-08,
        1.783e-08,  1.888e-08,  2.000e-08,  2.114e-08,  2.246e-08,
        2.374e-08,  2.498e-08,  2.631e-08,  2.760e-08,  2.899e-08,
        3.043e-08,  3.188e-08,  3.349e-08,  3.543e-08,  3.732e-08,
        3.912e-08,  4.081e-08,  4.252e-08,  4.454e-08,  4.628e-08,
        4.784e-08,  4.935e-08,  5.103e-08,  5.292e-08,  5.486e-08,
        5.702e-08,  5.920e-08,  6.045e-08,  6.317e-08,  6.630e-08,
        6.945e-08,  7.282e-08,  7.599e-08,  8.020e-08,  8.432e-08,
        8.814e-08,  9.196e-08,  9.582e-08,  9.978e-08,  1.038e-07,
        1.078e-07,  1.115e-07,  1.143e-07,  1.182e-07,  1.222e-07,
        1.264e-07,  1.307e-07,  1.349e-07,  1.389e-07,  1.432e-07,
        1.488e-07,  1.543e-07,  1.596e-07,  1.650e-07,  1.702e-07,
        1.753e-07,  1.795e-07,  1.835e-07,  1.885e-07,  1.945e-07,
        2.005e-07,  2.056e-07,  2.103e-07,  2.149e-07,  2.195e-07,
        2.234e-07,  2.267e-07,  2.287e-07,  2.310e-07,  2.322e-07,
        2.335e-07,  2.346e-07,  2.349e-07,  2.345e-07,  2.345e-07,
        2.338e-07,  2.328e-07,  2.313e-07,  2.290e-07,  2.262e-07,
        2.221e-07,  2.164e-07,  2.094e-07,  1.993e-07,  1.866e-07,
        1.680e-07,  1.478e-07,  1.301e-07,  1.210e-07,  1.176e-07,
        1.186e-07,  1.231e-07,  1.292e-07,  1.386e-07,  1.562e-07,
        1.758e-07,  1.934e-07,  2.154e-07,  2.415e-07,  2.582e-07,
        2.794e-07,  2.968e-07,  3.101e-07,  3.216e-07,  3.337e-07,
        3.480e-07,  3.600e-07,  3.730e-07,  3.840e-07,  3.930e-07,
        4.000e-07,  4.040e-07,  4.080e-07,  4.111e-07,  4.128e-07,
        4.127e-07,  4.116e-07,  4.088e-07,  4.021e-07,  3.931e-07,
        3.782e-07,  3.619e-07,  3.457e-07,  3.263e-07,  3.031e-07,
        2.806e-07,  2.633e-07,  2.461e-07,  2.288e-07,  2.147e-07,
        1.980e-07,  1.806e-07,  1.535e-07,  1.331e-07,  1.039e-07,
        6.677e-08,  4.973e-08,  4.163e-08,  3.696e-08,  3.430e-08,
        3.243e-08,  2.983e-08,  2.845e-08,  2.709e-08,  2.574e-08,
        2.438e-08,  2.342e-08,  2.273e-08,  2.228e-08,  2.197e-08,
        2.167e-08,  2.134e-08,  2.079e-08,  2.011e-08,  1.942e-08,
        1.938e-08,  1.933e-08,  1.927e-08,  1.922e-08,  1.917e-08,
        1.912e-08,  1.907e-08,  1.902e-08,  1.897e-08,  1.892e-08,
        1.887e-08,  1.882e-08,  1.877e-08,  1.872e-08,  1.867e-08,
        1.862e-08,  1.857e-08,  1.852e-08,  1.847e-08,  1.842e-08,
        1.837e-08,  1.832e-08,  1.827e-08,  1.822e-08,  1.817e-08,
        1.812e-08,  1.807e-08,  1.802e-08,  1.797e-08,  1.792e-08,
        1.787e-08,  1.782e-08,  1.777e-08,  1.772e-08,  1.767e-08,
        0.000e+00
    ])
    
    V1S = bo2inf3_nu_list.min()
    V2S = bo2inf3_nu_list.max()
    DVS = bo2inf3_nu_list[1] - bo2inf3_nu_list[0]
    NPTS = len(bo2inf3_nu_list)
    xo2inf3 = bo2inf3

    
    dvc = DVS
    v1c = v1abs - dvc
    v2c = v2abs + dvc

    i1 = -1 if v1c < V1S else int((v1c - V1S)/DVS + 0.01)
    v1c = V1S + DVS*(i1 - 1)
    i2 = int((v2c - V1S)/DVS + 0.01)
    nptc = i2 - i1 + 3
    if nptc > NPTS:
        nptc = NPTS + 4
    v2c = v1c + DVS*(nptc - 1)

    C = np.zeros(nptc)
    for j in range(nptc):
        idx = i1 + j
        if 1 <= idx <= NPTS:
            vj = v1c + dvc*j
            C[j] = xo2inf3[idx-1] / vj
    freqs = v1c + dvc*np.arange(nptc)
    return freqs, C, v1c, v2c, dvc, nptc

def o2_collision_profile(abs_v1, abs_v2, abs_dv,
                   p_arr, T_arr, thickness_arr,
                   air_arr,
                   xO2_arr, xN2_arr, xH2O_arr,
                   RHOAVE_arr, XKT_arr, amagat_arr,
                   JRAD=False
                    ):
    """ Compute and merge all O2 Collision Induced. """
    nlev = p_arr.size
    npt_abs = int((abs_v2 - abs_v1) / abs_dv) + 1
    C_o2_collision_all = np.zeros((nlev, npt_abs), dtype=float)

    # 1) Mate et al.
    #
    # O2 continuum formulated by Mate et al. over the spectral region
    # 7550-8486 cm-1:  "Absolute Intensities for the O2 1.27 micron
    # continuum absorption", B. Mate, C. Lugez, G.T. Fraser, and
    # W.J. Lafferty, J. Geophys. Res., 104, 30,585-30,590, 1999.
    #
    # Units of these coefficients are 1 / (amagat_O2*amagat_air)
    #
    # Also, refer to the paper "Observed  Atmospheric
    # Collision Induced Absorption in Near Infrared Oxygen Bands",
    # Mlawer, Clough, Brown, Stephen, Landry, Goldman, & Murcray,
    # Journal of Geophysical Research (1998).
    #
    # Only calculate if V2 > 7536. cm-1 and V1 <  8500. cm-1
    
    if abs_v2 > 7536 and abs_v1 < 8500 and XO2CN>0:
        a_o2, a_n2, a_h2o = 1/0.446, 0.3/0.446, 1.0
        for i in range(nlev):
            WKO2 = air_arr[i]*xO2_arr[i] # in moles/cm3
            WKO2_col = WKO2 * (thickness_arr[i]*1000*100) # in moles/cm2
            tau_fac = XO2CN*(WKO2_col/XLOSMT)*amagat_arr[i]*(a_o2*xO2_arr[i] + a_n2*xN2_arr[i] + a_h2o*xH2O_arr[i])
            freqs1, C0, v1c, v2c, dvc, nptc_o2inf1 = o2inf1(abs_v1, abs_v2)
            C1 = tau_fac * C0
            if JRAD==1:
                C1 *= np.array([radfn_cal(v, XKT_arr[i]) for v in freqs1])
                
            i_lo, i_hi, frac = pre_xint(
                cont_v1=v1c, cont_v2=v2c, cont_dv=dvc, npt_cont=nptc_o2inf2,
                abs_v1=abs_v1, abs_v2=abs_v2, abs_dv=abs_dv, npt_abs=npt_abs
            )
            abs_array = np.zeros(npt_abs, dtype=float)
            xint(cont_values=C1, i_lo=i_lo, i_hi=i_hi, frac=frac, abs_array=abs_array)
            
            C_o2_collision_all[i, :] = abs_array

    # 2) Mlawer IR
    #
    # O2 continuum formulated by Mlawer et al. over the spectral
    # region 9100-11000 cm-1. Refer to the paper "Observed
    # Atmospheric Collision Induced Absorption in Near Infrared
    # Oxygen Bands", Mlawer, Clough, Brown, Stephen, Landry, Goldman,
    # & Murcray, Journal of Geophysical Research (1998).
    # 
    # Only calculate if V2 > 9100. cm-1 and V1 <  11000. cm-1
    
    elif abs_v2 > 9100 and abs_v1 < 11000 and XO2CN>0:
        for i in range(nlev):
            freqs2, C2, v1c, v2c, dvc, nptc_o2inf2 = o2inf2(abs_v1, abs_v2)
            WKO2 = air_arr[i]*xO2_arr[i] # in moles/cm3
            WKO2_col = WKO2 * (thickness_arr[i]*1000*100) # in moles/cm2
            WO2    = XO2CN*(WKO2_col*WK_CONST)*RHOAVE_arr[i]
            ADJWO2 = xO2_arr[i]*(1/0.209)*WO2
            C2 = C2 * ADJWO2
            if JRAD==1:
                C2 *= np.array([radfn_cal(v, XKT_arr[i]) for v in freqs2])
            i_lo, i_hi, frac = pre_xint(
                cont_v1=v1c, cont_v2=v2c, cont_dv=dvc, npt_cont=nptc_o2inf2,
                abs_v1=abs_v1, abs_v2=abs_v2, abs_dv=abs_dv, npt_abs=npt_abs
            )
            abs_array = np.zeros(npt_abs, dtype=float)
            xint(cont_values=C2, i_lo=i_lo, i_hi=i_hi, frac=frac, abs_array=abs_array)
            
            C_o2_collision_all[i, :] = abs_array

    # 3) Mlawer A-band
    #  O2 A-band continuum formulated by Mlawer based on solar FTS measurements.
    #
    #    Only calculate if V2 > 12961.5 cm-1 and V1 < 13221.5 cm-1
    
    elif abs_v2 > 12961.5 and abs_v1 < 13221.5 and XO2CN>0:
        for i in range(nlev):
            # tau_fac similar to Mate’s first block
            WKO2 = air_arr[i]*xO2_arr[i] # in moles/cm3
            WKO2_col = WKO2 * (thickness_arr[i]*1000*100) # in moles/cm2
            tau_fac = XO2CN*(WKO2_col/XLOSMT)*amagat_arr[i]
            freqs3, C0_3,  v1c, v2c, dvc, nptc_o2inf3 = o2inf3(abs_v1, abs_v2)
            C3 = tau_fac * C0_3
            if JRAD==1:
                C3 *= np.array([radfn_cal(v, XKT_arr[i]) for v in freqs3])
            i_lo, i_hi, frac = pre_xint(
                cont_v1=v1c, cont_v2=v2c, cont_dv=dvc, npt_cont=nptc_o2inf3,
                abs_v1=abs_v1, abs_v2=abs_v2, abs_dv=abs_dv, npt_abs=npt_abs
            )
            abs_array = np.zeros(npt_abs, dtype=float)
            xint(cont_values=C3, i_lo=i_lo, i_hi=i_hi, frac=frac, abs_array=abs_array)
            
            C_o2_collision_all[i, :] = abs_array


        
    return C_o2_collision_all



# ─── Greenblatt et al. visible continuum ────────────────────────────────
def o2_vis(v1abs, v2abs):
    
    bo2_vis_data = pd.read_csv('bo2in_vis.csv')
    bo2_vis_nu = bo2_vis_data['x']
    bo2_vis_coeff = bo2_vis_data['f']
    
    V1S = bo2_vis_nu.min()
    V2S = bo2_vis_nu.max()
    DVS = bo2_vis_nu[1] - bo2_vis_nu[0]
    NPTS = len(bo2_vis_nu)
    
    factor = 1.0 / ((XLOSMT * WK_CONST * (55.0 * 273.0/296.0)**2) * 89.5)
    dvc = DVS
    v1c = v1abs - dvc
    v2c = v2abs + dvc
    i1 = -1 if v1c < V1S else int((v1c - V1S)/DVS + 0.01)
    v1c = V1S + DVS*(i1 - 1)
    i2 = int((v2c - V1S)/DVS + 0.01)
    nptc = i2 - i1 + 3
    if nptc > NPTS:
        nptc = NPTS + 4
    freqs = v1c + dvc * np.arange(nptc)
    C = np.zeros(nptc)
    for j in range(nptc):
        idx = i1 + j
        if 1 <= idx <= NPTS:
            vj = freqs[j]
            C[j] = factor * bo2_vis_coeff[idx-1] / vj
    return freqs, C, v1c, v2c, dvc, nptc

# ─── Herzberg UV continuum ─────────────────────────────────────────────
def hertda(vj: float) -> float:
    if vj <= 36000.0:
        return 0.0
    corr = ((40000.0 - vj)/4000.0)*7.917e-07 if vj <= 40000.0 else 0.0
    y = vj/48811.0
    return 6.884e-04*y*np.exp(-69.738*(np.log(y))**2) - corr

def herprs(herz: float, T: float, P: float) -> float:
    PO, TO = 1013.0, 273.16
    return herz * (1.0 + 0.83*(P/PO)*(TO/T))

def o2_herz(v1abs, v2abs, T, P):
    V1S, DVS = 36000.0, 10.0
    dvc = DVS
    v1c = v1abs - dvc
    v2c = v2abs + dvc
    i1 = -1 if v1c < V1S else int((v1c - V1S)/DVS + 0.01)
    v1c = V1S + DVS*(i1 - 1)
    i2 = int((v2c - V1S)/DVS + 0.01)
    nptc = i2 - i1 + 3
    freqs = v1c + dvc * np.arange(nptc)
    C = np.zeros(nptc)
    for j, vj in enumerate(freqs):
        H0 = hertda(vj)
        Hp = herprs(H0, T, P)
        C[j] = Hp / vj
    return freqs, C, v1c, v2c, dvc, nptc

# ─── Far‐UV Schumann–Runge continuum ────────────────────────────────────
def o2_fuv(v1abs, v2abs):
    
    # read the far‐UV Schumann–Runge continuum data
    # from the file bo2in_fuv.csv
    bo2in_fuv_data = pd.read_csv('bo2in_fuv.csv', header=None)
    bo2in_fuv_nu = bo2in_fuv_data['x'].values
    s_fuv = bo2in_fuv_data['f'].values
    V1S = bo2in_fuv_nu.min()
    V2S = bo2in_fuv_nu.max()
    DVS = bo2in_fuv_nu[1] - bo2in_fuv_nu[0]
    NPTS = len(bo2in_fuv_nu)
    
    dvc = DVS
    v1c = v1abs - dvc
    v2c = v2abs + dvc
    i1 = -1 if v1c < V1S else int((v1c - V1S)/DVS + 1e-5)
    v1c = V1S + DVS*(i1 - 1)
    i2 = int((v2c - V1S)/DVS + 1e-5)
    nptc = i2 - i1 + 3
    if nptc > NPTS:
        nptc = NPTS + 4
    freqs = v1c + dvc * np.arange(nptc)
    C = np.zeros(nptc)
    for j in range(nptc):
        idx = i1 + j
        if 1 <= idx <= NPTS:
            vj = freqs[j]
            C[j] = s_fuv[idx-1] / vj
    return freqs, C, v1c, v2c, dvc, nptc

# ─── Master routine to assemble everything and dump CSV ────────────────
def o2_continuum_profile(abs_v1, abs_v2, abs_dv,
                   p_arr, T_arr, thickness_arr,
                   air_arr, xO2_arr,
                   RHOAVE_arr, XKT_arr, amagat_arr,
                   JRAD=False
                    ):
    nlev = p_arr.size
    npt_abs = int((abs_v2 - abs_v1) / abs_dv) + 1
    C_o2_collision_all = np.zeros((nlev, npt_abs), dtype=float)
    
    # — Greenblatt vis
    if abs_v2 > 15000 and abs_v1 < 29870 and XO2CN>0:
        for i in range(nlev):
            WKO2 = xO2_arr[i]*air_arr[i] # in moles/cm3
            WKO2_col = WKO2 * (thickness_arr[i]*1000*100) # in moles/cm2
            WO2 = WKO2_col*WK_CONST*((p_arr[i]/1013.)*(273./T_arr[i]))*XO2CN
            CHIO2 = xO2_arr[i]
            ADJ = CHIO2 * WO2
            freq, C0, v1c, v2c, dvc, nptc_o2_vis = o2_vis(abs_v1, abs_v2)
            C = C0 * ADJ
            if JRAD==1:
                C *= radfn_cal(freq,XKT_arr[i])
            
            i_lo, i_hi, frac = pre_xint(
                cont_v1=v1c, cont_v2=v2c, cont_dv=dvc, npt_cont=nptc_o2_vis,
                abs_v1=abs_v1, abs_v2=abs_v2, abs_dv=abs_dv, npt_abs=npt_abs
            )
            abs_array = np.zeros(npt_abs, dtype=float)
            xint(cont_values=C, i_lo=i_lo, i_hi=i_hi, frac=frac, abs_array=abs_array)
            
            C_o2_collision_all[i, :] = abs_array

    # — Herzberg
    if abs_v2 > 36000 and XO2CN>0:
        for i in range(nlev):
            WKO2 = xO2_arr[i]*air_arr[i] # in moles/cm3
            WKO2_col = WKO2 * (thickness_arr[i]*1000*100) # in moles/cm2
            WO2 = WKO2_col * WK_CONST * XO2CN
            freq, C0, v1c, v2c, dvc, nptc_o2_herz = o2_herz(abs_v1, abs_v2, T_arr[i], p_arr[i])
            C = C0 * WO2
            if JRAD==1:
                C *= radfn_cal(freq,XKT_arr[i])
            
            i_lo, i_hi, frac = pre_xint(
                cont_v1=v1c, cont_v2=v2c, cont_dv=dvc, npt_cont=nptc_o2_herz,
                abs_v1=abs_v1, abs_v2=abs_v2, abs_dv=abs_dv, npt_abs=npt_abs
            )
            abs_array = np.zeros(npt_abs, dtype=float)
            xint(cont_values=C, i_lo=i_lo, i_hi=i_hi, frac=frac, abs_array=abs_array)
            
            C_o2_collision_all[i, :] = abs_array

    # — far‐UV
    if abs_v2 > 56740 and XO2CN>0:
        for i in range(nlev):
            WKO2 = xO2_arr[i]*air_arr[i] # in moles/cm3
            WKO2_col = WKO2 * (thickness_arr[i]*1000*100) # in moles/cm2
            WO2 = WKO2_col * WK_CONST * XO2CN
            freq, C0, v1c, v2c, dvc, nptc_o2_fuv = o2_fuv(abs_v1, abs_v2)
            C = C0 * WO2
            if JRAD==1:
                C *= radfn_cal(freq,XKT_arr[i])
                
            i_lo, i_hi, frac = pre_xint(
                cont_v1=v1c, cont_v2=v2c, cont_dv=dvc, npt_cont=nptc_o2_fuv,
                abs_v1=abs_v1, abs_v2=abs_v2, abs_dv=abs_dv, npt_abs=npt_abs
            )
            abs_array = np.zeros(npt_abs, dtype=float)
            xint(cont_values=C, i_lo=i_lo, i_hi=i_hi, frac=frac, abs_array=abs_array)
            
            C_o2_collision_all[i, :] = abs_array

    return C_o2_collision_all


def xn2_r(v1abs, v2abs, Tave):
    """
    
    ******** NITROGEN COLLISION INDUCED PURE ROTATION BAND  ********

        Model used:
         Borysow, A, and L. Frommhold, "Collision-induced
            rototranslational absorption spectra of N2-N2
            pairs for temperatures from 50 to 300 K", The
            Astrophysical Journal, 311, 1043-1057, 1986.

     Uodated 2004/09/22 based on:

      Boissoles, J., C. Boulet, R.H. Tipping, A. Brown and Q. Ma,
         Theoretical Calculations of the Translation-Rotation
         Collision-Induced Absorption in N2-N2, O2-O2 and N2-O2 Pairs,
         J.Quant. Spec. Rad. Transfer, 82,505 (2003).

         The temperature dependence between the two reference
         temperatures has been assumed the same as that for the
         original continuum.

        THIS NITROGEN CONTINUUM IS IN UNITS OF 1./(CM AMAGAT^2)

        Only calculate if V2 > -10. cm-1 and V1 <  350. cm-1
    
    Rototranslational N2–N2 continuum, port of FORTRAN xn2_r.
    Returns (freqs, c0, c1)
    """
    
    bn2t296_nu = np.arange(-10.,  350.+1e-7,  5)


    bn2t296_CT296 = np.array([
        0.4303e-06, 0.4850e-06, 0.4979e-06, 0.4850e-06, 0.4303e-06,
        0.3715e-06, 0.3292e-06, 0.3086e-06, 0.2920e-06, 0.2813e-06,
        0.2804e-06, 0.2738e-06, 0.2726e-06, 0.2724e-06, 0.2635e-06,
        0.2621e-06, 0.2547e-06, 0.2428e-06, 0.2371e-06, 0.2228e-06,
        0.2100e-06, 0.1991e-06, 0.1822e-06, 0.1697e-06, 0.1555e-06,
        0.1398e-06, 0.1281e-06, 0.1138e-06, 0.1012e-06, 0.9078e-07,
        0.7879e-07, 0.6944e-07, 0.6084e-07, 0.5207e-07, 0.4540e-07,
        0.3897e-07, 0.3313e-07, 0.2852e-07, 0.2413e-07, 0.2045e-07,
        0.1737e-07, 0.1458e-07, 0.1231e-07, 0.1031e-07, 0.8586e-08,
        0.7162e-08, 0.5963e-08, 0.4999e-08, 0.4226e-08, 0.3607e-08,
        0.3090e-08, 0.2669e-08, 0.2325e-08, 0.2024e-08, 0.1783e-08,
        0.1574e-08, 0.1387e-08, 0.1236e-08, 0.1098e-08, 0.9777e-09,
        0.8765e-09, 0.7833e-09, 0.7022e-09, 0.6317e-09, 0.5650e-09,
        0.5100e-09, 0.4572e-09, 0.4115e-09, 0.3721e-09, 0.3339e-09,
        0.3005e-09, 0.2715e-09, 0.2428e-09
    ])

    bn2t296_sf296 = np.array([
        1.3534, 1.3517, 1.3508, 1.3517, 1.3534,
        1.3558, 1.3584, 1.3607, 1.3623, 1.3632,
        1.3634, 1.3632, 1.3627, 1.3620, 1.3612,
        1.3605, 1.3597, 1.3590, 1.3585, 1.3582,
        1.3579, 1.3577, 1.3577, 1.3580, 1.3586,
        1.3594, 1.3604, 1.3617, 1.3633, 1.3653,
        1.3677, 1.3706, 1.3742, 1.3780, 1.3822,
        1.3868, 1.3923, 1.3989, 1.4062, 1.4138,
        1.4216, 1.4298, 1.4388, 1.4491, 1.4604,
        1.4718, 1.4829, 1.4930, 1.5028, 1.5138,
        1.5265, 1.5392, 1.5499, 1.5577, 1.5639,
        1.5714, 1.5816, 1.5920, 1.6003, 1.6051,
        1.6072, 1.6097, 1.6157, 1.6157, 1.6157,
        1.6157, 1.6157, 1.6157, 1.6157, 1.6157,
        1.6157, 1.6157, 1.6157
    ])


    bn2t220_CT220 = np.array([
        0.4946E-06, 0.5756E-06, 0.5964E-06, 0.5756E-06, 0.4946E-06,
        0.4145E-06, 0.3641E-06, 0.3482E-06, 0.3340E-06, 0.3252E-06,
        0.3299E-06, 0.3206E-06, 0.3184E-06, 0.3167E-06, 0.2994E-06,
        0.2943E-06, 0.2794E-06, 0.2582E-06, 0.2468E-06, 0.2237E-06,
        0.2038E-06, 0.1873E-06, 0.1641E-06, 0.1474E-06, 0.1297E-06,
        0.1114E-06, 0.9813E-07, 0.8309E-07, 0.7059E-07, 0.6068E-07,
        0.5008E-07, 0.4221E-07, 0.3537E-07, 0.2885E-07, 0.2407E-07,
        0.1977E-07, 0.1605E-07, 0.1313E-07, 0.1057E-07, 0.8482E-08,
        0.6844E-08, 0.5595E-08, 0.4616E-08, 0.3854E-08, 0.3257E-08,
        0.2757E-08, 0.2372E-08, 0.2039E-08, 0.1767E-08, 0.1548E-08,
        0.1346E-08, 0.1181E-08, 0.1043E-08, 0.9110E-09, 0.8103E-09,
        0.7189E-09, 0.6314E-09, 0.5635E-09, 0.4976E-09, 0.4401E-09,
        0.3926E-09, 0.3477E-09, 0.3085E-09, 0.2745E-09, 0.2416E-09,
        0.2155E-09, 0.1895E-09, 0.1678E-09, 0.1493E-09, 0.1310E-09,
        0.1154E-09, 0.1019E-09, 0.8855E-10
    ])

    bn2t220_sf220 = np.array([
        1.3536,     1.3515,     1.3502,     1.3515,     1.3536,
        1.3565,     1.3592,     1.3612,     1.3623,     1.3626,
        1.3623,     1.3616,     1.3609,     1.3600,     1.3591,
        1.3583,     1.3576,     1.3571,     1.3571,     1.3572,
        1.3574,     1.3578,     1.3585,     1.3597,     1.3616,
        1.3640,     1.3666,     1.3698,     1.3734,     1.3776,
        1.3828,     1.3894,     1.3969,     1.4049,     1.4127,
        1.4204,     1.4302,     1.4427,     1.4562,     1.4687,
        1.4798,     1.4894,     1.5000,     1.5142,     1.5299,
        1.5441,     1.5555,     1.5615,     1.5645,     1.5730,
        1.5880,     1.6028,     1.6121,     1.6133,     1.6094,
        1.6117,     1.6244,     1.6389,     1.6485,     1.6513,
        1.6468,     1.6438,     1.6523,     1.6523,     1.6523,
        1.6523,     1.6523,     1.6523,     1.6523,     1.6523,
        1.6523,     1.6523,     1.6523
    ])
    
    V1S = bn2t296_nu.min()
    V2S = bn2t296_nu.max()
    DVS = bn2t296_nu[1] - bn2t296_nu[0]
    NPTS = len(bn2t296_nu)
    
    # mixing‐ratio constants
    xo2 = 0.21
    xn2 = 0.79
    T_296, T_220 = 296.0, 220.0

    tfac = (Tave - T_296) / (T_220 - T_296)

    dvc = DVS
    v1c = v1abs - dvc
    v2c = v2abs + dvc
    i1 = -1 if v1c < V1S else int((v1c - V1S) / DVS + 0.01)
    v1c = V1S + DVS * (i1 - 1)
    i2 = int((v2c - V1S) / DVS + 0.01)
    nptc = i2 - i1 + 3
    if nptc > NPTS:
        nptc = NPTS + 4
    freqs = v1c + dvc * np.arange(nptc)

    c0 = np.zeros(nptc)
    c1 = np.zeros(nptc)
    for j in range(nptc):
        idx = i1 + j
        if 1 <= idx <= NPTS:
            # interpolate C_296→C_220 logarithmically
            c296 = bn2t296_CT296[idx-1]
            c220 = bn2t220_CT220[idx-1]
            val = c296 * (c220 / c296)**tfac if (c296 > 0 and c220 > 0) else c296 + (c220 - c296)*( (Tave-T_296)/(T_220-T_296) )
            c0[j] = val
            # scale‐factor interpolation
            sf296 = bn2t296_sf296[idx-1]
            sf220 = bn2t220_sf220[idx-1]
            sfT = sf296 * (sf220 / sf296)**tfac
            c1[j] = (sfT - 1.) * (xn2 / xo2)
    return freqs, c0, c1, v1c, v2c, dvc, nptc

def n2_ver_1(v1abs, v2abs, Tave):
    """
    N2 fundamental near 4.3 µm, port of FORTRAN n2_ver_1.
    Returns (freqs, cn0, cn1, cn2)
    """
    
    bn2f_nu_list = np.arange(1997.784896, 2901.576661+1e-7, 3.981461525)

    bn2f_xn2_272 = np.array([
        0.000E+00,
        4.691E-11,  5.960E-11,  7.230E-11,  9.435E-11,  1.171E-10,
        1.472E-10,  1.874E-10,  2.276E-10,  2.960E-10,  3.671E-10, 
        4.605E-10,  5.874E-10,  7.144E-10,  9.293E-10,  1.155E-09, 
        1.447E-09,  1.847E-09,  2.247E-09,  2.919E-09,  3.635E-09, 
        4.594E-09,  6.003E-09,  7.340E-09,  9.190E-09,  1.130E-08, 
        1.370E-08,  1.650E-08,  1.960E-08,  2.310E-08,  2.710E-08, 
        3.160E-08,  3.660E-08,  4.230E-08,  4.860E-08,  5.570E-08, 
        6.350E-08,  7.230E-08,  8.200E-08,  9.270E-08,  1.050E-07, 
        1.180E-07,  1.320E-07,  1.480E-07,  1.650E-07,  1.840E-07, 
        2.040E-07,  2.270E-07,  2.510E-07,  2.770E-07,  3.060E-07, 
        3.360E-07,  3.670E-07,  4.010E-07,  4.330E-07,  4.710E-07, 
        5.050E-07,  5.450E-07,  5.790E-07,  6.200E-07,  6.540E-07, 
        6.940E-07,  7.240E-07,  7.610E-07,  7.880E-07,  8.220E-07, 
        8.440E-07,  8.720E-07,  8.930E-07,  9.190E-07,  9.370E-07, 
        9.620E-07,  9.870E-07,  1.020E-06,  1.060E-06,  1.110E-06, 
        1.180E-06,  1.280E-06,  1.400E-06,  1.570E-06,  1.750E-06, 
        1.880E-06,  2.020E-06,  2.080E-06,  2.060E-06,  1.960E-06, 
        1.860E-06,  1.710E-06,  1.570E-06,  1.490E-06,  1.440E-06, 
        1.410E-06,  1.390E-06,  1.380E-06,  1.380E-06,  1.390E-06, 
        1.390E-06,  1.410E-06,  1.420E-06,  1.430E-06,  1.420E-06, 
        1.430E-06,  1.410E-06,  1.400E-06,  1.370E-06,  1.350E-06, 
        1.310E-06,  1.270E-06,  1.220E-06,  1.170E-06,  1.120E-06, 
        1.060E-06,  1.010E-06,  9.470E-07,  8.910E-07,  8.290E-07, 
        7.740E-07,  7.160E-07,  6.620E-07,  6.090E-07,  5.600E-07, 
        5.130E-07,  4.680E-07,  4.290E-07,  3.900E-07,  3.560E-07, 
        3.240E-07,  2.950E-07,  2.680E-07,  2.440E-07,  2.230E-07, 
        2.030E-07,  1.850E-07,  1.690E-07,  1.540E-07,  1.410E-07, 
        1.290E-07,  1.180E-07,  1.080E-07,  9.950E-08,  9.100E-08, 
        8.380E-08,  7.700E-08,  7.100E-08,  6.510E-08,  6.010E-08, 
        5.550E-08,  5.110E-08,  4.710E-08,  4.340E-08,  3.980E-08, 
        3.660E-08,  3.380E-08,  3.110E-08,  2.840E-08,  2.610E-08, 
        2.390E-08,  2.210E-08,  2.010E-08,  1.830E-08,  1.710E-08, 
        1.550E-08,  1.450E-08,  1.320E-08,  1.208E-08,  1.112E-08, 
        1.015E-08,  9.339E-09,  8.597E-09,  7.873E-09,  7.247E-09, 
        6.620E-09,  6.074E-09,  5.570E-09,  5.081E-09,  4.676E-09, 
        4.272E-09,  3.919E-09,  3.595E-09,  3.279E-09,  3.019E-09, 
        2.758E-09,  2.529E-09,  2.320E-09,  2.115E-09,  1.948E-09, 
        1.780E-09,  1.632E-09,  1.497E-09,  1.365E-09,  1.257E-09, 
        1.149E-09,  1.053E-09,  9.663E-10,  8.806E-10,  8.111E-10, 
        7.416E-10,  6.795E-10,  6.237E-10,  5.682E-10,  5.233E-10, 
        4.785E-10,  4.383E-10,  4.024E-10,  3.666E-10,  3.378E-10, 
        3.090E-10,  2.829E-10,  2.598E-10,  2.366E-10,  2.180E-10, 
        1.994E-10,  1.825E-10,  1.676E-10,  1.527E-10,  1.406E-10, 
        1.287E-10,  1.178E-10,  1.082E-10,  9.859E-11,  9.076E-11, 
        8.305E-11,  7.599E-11,  6.981E-11,  6.363E-11,  5.857E-11, 
        5.362E-11,  0.000E+00])

    bn2f_xn2_228 = np.array([
        0.000E+00,
        5.736E-11,  7.296E-11,  8.856E-11,  1.154E-10,  1.431E-10,
        1.799E-10,  2.291E-10,  2.783E-10,  3.623E-10,  4.497E-10,
        5.642E-10,  7.195E-10,  8.749E-10,  1.137E-09,  1.413E-09,
        1.769E-09,  2.259E-09,  2.749E-09,  3.568E-09,  4.440E-09,
        5.549E-09,  7.097E-09,  8.645E-09,  1.120E-08,  1.395E-08,
        1.650E-08,  1.880E-08,  2.130E-08,  2.400E-08,  2.690E-08,
        3.010E-08,  3.360E-08,  3.750E-08,  4.180E-08,  4.670E-08,
        5.210E-08,  5.830E-08,  6.520E-08,  7.290E-08,  8.170E-08,
        9.150E-08,  1.030E-07,  1.150E-07,  1.290E-07,  1.440E-07,
        1.610E-07,  1.800E-07,  2.020E-07,  2.250E-07,  2.510E-07,
        2.790E-07,  3.090E-07,  3.430E-07,  3.770E-07,  4.160E-07,
        4.540E-07,  4.990E-07,  5.370E-07,  5.850E-07,  6.250E-07,
        6.750E-07,  7.130E-07,  7.610E-07,  7.970E-07,  8.410E-07,
        8.720E-07,  9.100E-07,  9.380E-07,  9.720E-07,  9.940E-07,
        1.020E-06,  1.050E-06,  1.080E-06,  1.120E-06,  1.170E-06,
        1.240E-06,  1.340E-06,  1.470E-06,  1.660E-06,  1.870E-06,
        2.040E-06,  2.220E-06,  2.300E-06,  2.290E-06,  2.160E-06,
        2.050E-06,  1.870E-06,  1.710E-06,  1.620E-06,  1.580E-06,
        1.550E-06,  1.540E-06,  1.540E-06,  1.550E-06,  1.560E-06,
        1.570E-06,  1.590E-06,  1.590E-06,  1.600E-06,  1.580E-06,
        1.570E-06,  1.540E-06,  1.510E-06,  1.470E-06,  1.430E-06,
        1.370E-06,  1.310E-06,  1.250E-06,  1.180E-06,  1.110E-06,
        1.040E-06,  9.740E-07,  9.020E-07,  8.360E-07,  7.650E-07,
        7.050E-07,  6.430E-07,  5.860E-07,  5.320E-07,  4.820E-07,
        4.370E-07,  3.950E-07,  3.570E-07,  3.220E-07,  2.910E-07,
        2.630E-07,  2.390E-07,  2.160E-07,  1.960E-07,  1.780E-07,
        1.620E-07,  1.480E-07,  1.330E-07,  1.220E-07,  1.120E-07,
        1.020E-07,  9.280E-08,  8.420E-08,  7.700E-08,  6.990E-08,
        6.390E-08,  5.880E-08,  5.380E-08,  4.840E-08,  4.380E-08,
        4.020E-08,  3.690E-08,  3.290E-08,  3.050E-08,  2.720E-08,
        2.490E-08,  2.260E-08,  2.020E-08,  1.810E-08,  1.620E-08,
        1.500E-08,  1.359E-08,  1.232E-08,  1.111E-08,  1.011E-08,
        9.115E-09,  8.273E-09,  7.497E-09,  6.753E-09,  6.148E-09,
        5.543E-09,  5.029E-09,  4.558E-09,  4.105E-09,  3.738E-09,
        3.371E-09,  3.057E-09,  2.771E-09,  2.494E-09,  2.272E-09,
        2.049E-09,  1.858E-09,  1.685E-09,  1.516E-09,  1.381E-09,
        1.246E-09,  1.129E-09,  1.024E-09,  9.215E-10,  8.396E-10,
        7.578E-10,  6.865E-10,  6.228E-10,  5.601E-10,  5.105E-10,
        4.609E-10,  4.173E-10,  3.786E-10,  3.403E-10,  3.102E-10,
        2.802E-10,  2.536E-10,  2.302E-10,  2.069E-10,  1.886E-10,
        1.704E-10,  1.542E-10,  1.400E-10,  1.257E-10,  1.147E-10,
        1.036E-10,  9.371E-11,  8.509E-11,  7.647E-11,  6.970E-11,
        6.298E-11,  5.695E-11,  5.172E-11,  4.650E-11,  4.237E-11,
        3.829E-11,  3.462E-11,  3.145E-11,  2.828E-11,  2.576E-11,
        2.329E-11,  2.104E-11,  1.912E-11,  1.720E-11,  1.566E-11,
        1.416E-11,  0.000E+00])
    
    bn2f_a_h2o = np.array([
        200.00,                                    
        199.95,  179.14,  158.32,  143.10,  128.29,
        115.21,  104.51,  93.816,  85.716,  77.876,
        70.838,  65.017,  59.196,  54.652,  50.272,
        46.284,  42.951,  39.619,  36.945,  34.377,
        32.009,  30.010,  28.012,  26.370,  24.797,
        23.332,  22.085,  20.839,  19.794,  18.795,
        17.855,  17.051,  16.247,  15.562,  14.909,
        14.290,  13.756,  13.223,  12.763,  12.325,
        11.906,  11.544,  11.182,  10.870,  10.575,
        10.292,  10.045,  9.7985,  9.5833,  9.3802,
        9.1847,  9.0175,  8.8503,  8.7025,  8.5632,
        8.4290,  8.3176,  8.2061,  8.1081,  8.0165,
        7.9282,  7.8565,  7.7848,  7.7236,  7.6678,
        7.6138,  7.5700,  7.5262,  7.4899,  7.4581,
        7.4271,  7.4032,  7.3794,  7.3603,  7.3444,
        7.3292,  7.3212,  7.3133,  7.3076,  7.3036,
        7.2997,  7.2957,  7.2917,  7.2900,  7.2900,
        7.2900,  7.2900,  7.2900,  7.2921,  7.2961,
        7.3000,  7.3000,  7.3000,  7.3040,  7.3120,
        7.3200,  7.3359,  7.3518,  7.3736,  7.4015,
        7.4293,  7.4728,  7.5166,  7.5697,  7.6334,
        7.6971,  7.7874,  7.8790,  7.9846,  8.1081,
        8.2315,  8.3920,  8.5552,  8.7419,  8.9609,
        9.1798,  9.4567,  9.7394,  10.059,  10.438,
        10.816,  11.300,  11.798,  12.358,  13.023,
        13.687,  14.541,  15.425,  16.402,  17.553,
        18.704,  20.098,  21.539,  23.061,  24.749,
        26.437,  28.236,  30.059,  31.710,  32.964,
        34.218,  35.570,  36.948,  38.366,  39.886,
        41.407,  43.045,  44.717,  46.432,  48.271,
        50.111,  52.090,  54.116,  56.190,  58.420,
        60.649,  63.040,  65.493,  67.997,  70.697,
        73.396,  76.287,  79.262,  82.291,  85.560,
        88.828,  92.320,  95.920,  99.578,  103.54,
        107.50,  111.72,  116.08,  120.50,  125.30,
        130.09,  135.20,  140.48,  145.83,  151.64,
        157.45,  163.62,  170.01,  176.47,  183.51,
        190.55,  198.01,  205.75,  213.56,  222.09,
        230.61,  239.62,  249.01,  258.43,  268.76,
        279.09,  289.98,  301.35,  312.75,  325.26,
        337.77,  350.92,  364.69,  378.46,  393.61,
        408.77,  424.67,  441.35,  458.03,  476.35,
        494.70,  513.92,  534.12,  554.33,  576.47,
        598.70,  621.92,  646.40,  670.87,  697.63,
        724.56,  752.63,  782.27,  811.91,  844.26,
        876.88,  876.88])
    
    
    
    V1S = bn2f_nu_list.min()
    V2S = bn2f_nu_list.max()
    DVS = bn2f_nu_list[1] - bn2f_nu_list[0]
    NPTS = len(bn2f_nu_list)
    
    T_272, T_228 = 272.0, 228.0
    xtfac = ((1/Tave)-(1/T_272))/((1/T_228)-(1/T_272))
    xt_lin = (Tave - T_272)/(T_228 - T_272)
    a_o2 = 1.294 - 0.4545 * Tave/296.0

    dvc = DVS
    v1c = v1abs - dvc
    v2c = v2abs + dvc
    i1 = -1 if v1c < V1S else int((v1c - V1S)/DVS + 0.01)
    v1c = V1S + DVS * (i1 - 1)
    i2 = int((v2c - V1S)/DVS + 0.01)
    nptc = i2 - i1 + 3
    if nptc > NPTS:
        nptc = NPTS + 4
    freqs = v1c + dvc * np.arange(nptc)

    cn0 = np.zeros(nptc)
    cn1 = np.zeros(nptc)
    cn2 = np.zeros(nptc)
    for j in range(nptc):
        idx = i1 + j
        if 1 <= idx <= NPTS:
            # temperature interpolation
            v272 = bn2f_xn2_272[idx-1]
            v228 = bn2f_xn2_228[idx-1]
            if v272>0 and v228>0:
                base = v272*(v228/v272)**xtfac
            else:
                base = v272 + (v228 - v272)*xt_lin
            vj = freqs[j]
            coef = base / vj
            cn0[j] = coef
            cn1[j] = a_o2 * coef
            cn2[j] = (9/7) * bn2f_a_h2o[idx-1] * coef
    return freqs, cn0, cn1, cn2, v1c, v2c, dvc, nptc

def n2_overtone1(v1abs, v2abs):
    """
    N2 first overtone, port of FORTRAN n2_overtone1.
    Returns (freqs, c0)
    """
    
    bn2f1_nu = np.arange(4340.0, 4910.0+1e-7, 3.0)

    bn2f1_xn2 = np.array([
        0.000E+00, 3.709E-11, 7.418E-11, 1.113E-10, 1.484E-10,
        1.843E-10, 2.163E-10, 2.482E-10, 2.802E-10, 3.122E-10,
        3.442E-10, 3.640E-10, 3.776E-10, 3.912E-10, 4.048E-10,
        4.183E-10, 4.334E-10, 4.626E-10, 4.918E-10, 5.210E-10,
        5.357E-10, 5.411E-10, 5.465E-10, 5.520E-10, 5.593E-10,
        5.853E-10, 6.114E-10, 6.375E-10, 6.635E-10, 6.855E-10,
        6.926E-10, 6.997E-10, 7.069E-10, 7.140E-10, 7.211E-10,
        7.283E-10, 7.380E-10, 7.551E-10, 7.722E-10, 7.893E-10,
        8.064E-10, 8.235E-10, 8.419E-10, 8.627E-10, 8.835E-10,
        9.043E-10, 9.251E-10, 9.779E-10, 1.066E-09, 1.154E-09,
        1.242E-09, 1.344E-09, 1.446E-09, 1.549E-09, 1.653E-09,
        1.759E-09, 1.865E-09, 1.977E-09, 2.103E-09, 2.228E-09,
        2.348E-09, 2.467E-09, 2.586E-09, 2.705E-09, 2.824E-09,
        2.944E-09, 3.066E-09, 3.188E-09, 3.309E-09, 3.426E-09,
        3.543E-09, 3.660E-09, 3.813E-09, 3.976E-09, 4.135E-09,
        4.309E-09, 4.499E-09, 4.700E-09, 4.905E-09, 5.105E-09,
        5.332E-09, 5.575E-09, 5.856E-09, 6.175E-09, 6.421E-09,
        6.640E-09, 7.086E-09, 7.508E-09, 7.906E-09, 8.304E-09,
        8.930E-09, 9.480E-09, 9.921E-09, 1.051E-08, 1.070E-08,
        1.090E-08, 1.090E-08, 1.090E-08, 1.090E-08, 1.070E-08,
        1.050E-08, 1.030E-08, 1.010E-08, 9.940E-09, 9.760E-09,
        9.580E-09, 9.400E-09, 9.220E-09, 9.040E-09, 8.860E-09,
        8.680E-09, 8.510E-09, 8.370E-09, 8.250E-09, 8.150E-09,
        8.070E-09, 8.010E-09, 7.950E-09, 7.890E-09, 7.830E-09,
        7.759E-09, 7.553E-09, 7.347E-09, 7.141E-09, 6.935E-09,
        6.729E-09, 6.523E-09, 6.317E-09, 6.111E-09, 5.905E-09,
        5.699E-09, 5.493E-09, 5.287E-09, 5.081E-09, 4.876E-09,
        4.670E-09, 4.464E-09, 4.258E-09, 4.052E-09, 3.846E-09,
        3.640E-09, 3.450E-09, 3.268E-09, 3.092E-09, 2.917E-09,
        2.741E-09, 2.566E-09, 2.392E-09, 2.219E-09, 2.076E-09,
        1.959E-09, 1.841E-09, 1.723E-09, 1.617E-09, 1.527E-09,
        1.437E-09, 1.346E-09, 1.256E-09, 1.172E-09, 1.093E-09,
        1.013E-09, 9.335E-10, 8.539E-10, 7.979E-10, 7.514E-10,
        7.050E-10, 6.586E-10, 6.121E-10, 5.687E-10, 5.413E-10,
        5.138E-10, 4.864E-10, 4.589E-10, 4.315E-10, 4.040E-10,
        3.770E-10, 3.504E-10, 3.238E-10, 2.972E-10, 2.706E-10,
        2.409E-10, 2.099E-10, 1.788E-10, 1.478E-10, 1.225E-10,
        1.021E-10, 8.165E-11, 6.123E-11, 4.082E-11, 2.041E-11,
        0.000E+00])
    
    V1S = bn2f1_nu.min()
    V2S = bn2f1_nu.max()
    DVS = bn2f1_nu[1] - bn2f1_nu[0]
    NPTS = len(bn2f1_nu)
    
    dvc = DVS
    v1c = v1abs - dvc
    v2c = v2abs + dvc
    i1 = -1 if v1c < V1S else int((v1c - V1S)/DVS + 0.01)
    v1c = V1S + DVS * (i1 - 1)
    i2 = int((v2c - V1S)/DVS + 0.01)
    nptc = i2 - i1 + 3
    if nptc > NPTS:
        nptc = NPTS + 4
    freqs = v1c + dvc * np.arange(nptc)

    c0 = np.zeros(nptc)
    for j in range(nptc):
        idx = i1 + j
        if 1 <= idx <= NPTS:
            c0[j] = bn2f1_xn2[idx-1]/freqs[j]
    return freqs, c0, v1c, v2c, dvc, nptc

def compute_n2_continuum(abs_v1, abs_v2, abs_dv,
                   p_arr, T_arr, thickness_arr,
                   air_arr, xO2_arr, xN2_arr, xH2O_arr,
                   RHOAVE_arr, XKT_arr, amagat_arr,
                   JRAD=False
                    ):

    nlev = p_arr.size
    npt_abs = int((abs_v2 - abs_v1) / abs_dv) + 1
    C_n2_continuum_all = np.zeros((nlev, npt_abs), dtype=float)

    # ── Rototranslational N2─N2 pure‐rotation
    if abs_v2 > -10 and abs_v1 < 350 and XN2CN > 0:
        for i in range(nlev):
            WKN2 = xN2_arr[i] * air_arr[i] # in molec/cm3
            WKN2_col = WKN2 * thickness_arr[i] * 1000 * 100 # in molec/cm2
            tau_fac = XN2CN * (WKN2_col/XLOSMT) * amagat_arr[i]
            freq, c0, c1, v1c, v2c, dvc, nptc_xn2_r = xn2_r(abs_v1, abs_v2, T_arr[i])
            C = tau_fac * c0 * (xN2_arr + c1*xO2_arr + 1*xH2O_arr)
            if JRAD == 1:
                C *= radfn_cal(freq, XKT_arr[i])
            
            i_lo, i_hi, frac = pre_xint(
                cont_v1=v1c, cont_v2=v2c, cont_dv=dvc, npt_cont=nptc_xn2_r,
                abs_v1=abs_v1, abs_v2=abs_v2, abs_dv=abs_dv, npt_abs=npt_abs
            )
            abs_array = np.zeros(npt_abs, dtype=float)
            xint(cont_values=C, i_lo=i_lo, i_hi=i_hi, frac=frac, abs_array=abs_array)
            
            C_n2_continuum_all[i, :] = abs_array

    # ── N2 fundamental
    if abs_v2 > 2001.77 and abs_v1 < 2897.59 and XN2CN > 0:
        for i in range(nlev):
            WKN2 = xN2_arr[i] * air_arr[i] # in molec/cm3
            WKN2_col = WKN2 * thickness_arr[i] * 1000 * 100 # in molec/cm2
            tau_fac = XN2CN * (WKN2_col/XLOSMT) * amagat_arr[i]
            freq, cn0, cn1, cn2, v1c, v2c, dvc, nptc_n2_ver_1 = n2_ver_1(abs_v1, abs_v2, T_arr[i])
            C = tau_fac * (xN2_arr*cn0 + xO2_arr*cn1 + xH2O_arr*cn2)
            if JRAD == 1:
                C *= radfn_cal(freq, XKT_arr[i])
            
            i_lo, i_hi, frac = pre_xint(
                cont_v1=v1c, cont_v2=v2c, cont_dv=dvc, npt_cont=nptc_n2_ver_1,
                abs_v1=abs_v1, abs_v2=abs_v2, abs_dv=abs_dv, npt_abs=npt_abs
            )
            abs_array = np.zeros(npt_abs, dtype=float)
            xint(cont_values=C, i_lo=i_lo, i_hi=i_hi, frac=frac, abs_array=abs_array)
            
            C_n2_continuum_all[i, :] = abs_array

    # ── N2 first overtone
    if abs_v2 > 4340 and abs_v1 < 4910 and XN2CN > 0:
        for i in range(nlev):
            WKN2 = xN2_arr[i] * air_arr[i] # in molec/cm3
            WKN2_col = WKN2 * thickness_arr[i] * 1000 * 100 # in molec/cm2
            tau_fac = XN2CN * (WKN2_col/XLOSMT) * amagat_arr[i] * \
                    (xN2_arr + 1*xO2_arr + 1*xH2O_arr)
            freq, c0, v1c, v2c, dvc, nptc_n2_overtone1 = n2_overtone1(abs_v1, abs_v2)
            C = tau_fac * c0
            if JRAD == 1:
                C *= radfn_cal(freq, XKT_arr[i])
                
            i_lo, i_hi, frac = pre_xint(
                cont_v1=v1c, cont_v2=v2c, cont_dv=dvc, npt_cont=nptc_n2_overtone1,
                abs_v1=abs_v1, abs_v2=abs_v2, abs_dv=abs_dv, npt_abs=npt_abs
            )
            abs_array = np.zeros(npt_abs, dtype=float)
            xint(cont_values=C, i_lo=i_lo, i_hi=i_hi, frac=frac, abs_array=abs_array)
            
            C_n2_continuum_all[i, :] = abs_array

    return C_n2_continuum_all


def compute_continuum_profile(
    abs_v1, abs_v2, abs_dv,
    p_arr, T_arr, thickness_lay, air_lay, 
    xCO2_arr, xN2_arr, xO2_arr, xH2O_arr, xO3_arr, 
    RHOAVE_arr, XKT_arr, amagat_lay,
    JRAD=False
    ):
    """
    Compute the total continuum absorption for a full vertical profile.
    """
    nlev = p_arr.size
    npt_abs = int((abs_v2 - abs_v1) / abs_dv) + 1
    nu_list = np.arange(abs_v1, abs_v2 + 1e-7, abs_dv)
    ABSRB_profile = np.zeros((nlev, npt_abs), dtype=float)

    # CO₂ continuum
    C_co2_all = frnco2_profile(
        abs_v1=abs_v1, abs_v2=abs_v2, abs_dv=abs_dv,
        p_arr=p_arr, T_arr=T_arr, thickness_arr=thickness_lay,
        air_arr=air_lay, xCO2_arr=xCO2_arr,
        RHOAVE_arr=RHOAVE_arr,
        XKT_arr=XKT_arr,
        JRAD=JRAD
    )
    
    C_o3_all, o3_diff_V1C, o3_diff_V2C, o3_diff_DVC, o3_diff_NPTO3 = o3_diffuse_profile(
        abs_v1=abs_v1, abs_v2=abs_v2, abs_dv=abs_dv,
        p_arr=p_arr, T_arr=T_arr, thickness_arr=thickness_lay,
        air_arr=air_lay, xO3_arr=xO3_arr, 
        RHOAVE_arr=RHOAVE_arr,
        XKT_arr=XKT_arr,
        JRAD=JRAD
    )
    
    C_o2_ver1_all = o2_ver1_profile(
        abs_v1=abs_v1, abs_v2=abs_v2, abs_dv=abs_dv,
        p_arr=p_arr, T_arr=T_arr, thickness_arr=thickness_lay,
        air_arr=air_lay, xO2_arr=xO2_arr, 
        RHOAVE_arr=RHOAVE_arr,
        XKT_arr=XKT_arr,
        amagat_arr=amagat_lay,
        JRAD=JRAD
    )
    
    C_o2_collision_all = o2_collision_profile(
        abs_v1=abs_v1, abs_v2=abs_v2, abs_dv=abs_dv,
        p_arr=p_arr, T_arr=T_arr, thickness_arr=thickness_lay,
        air_arr=air_lay,
        xO2_arr=xO2_arr, xN2_arr=xN2_arr, xH2O_arr=xH2O_arr,
        RHOAVE_arr=RHOAVE_arr, XKT_arr=XKT_arr, amagat_arr=amagat_lay,
        JRAD=JRAD
    )
    
    C_o2_continuum_all = o2_continuum_profile(
        abs_v1=abs_v1, abs_v2=abs_v2, abs_dv=abs_dv,
        p_arr=p_arr, T_arr=T_arr, thickness_arr=thickness_lay,
        air_arr=air_lay, xO2_arr=xO2_arr,
        RHOAVE_arr=RHOAVE_arr, XKT_arr=XKT_arr, amagat_arr=amagat_lay,
        JRAD=JRAD
    )
    
    C_n2_continuum = compute_n2_continuum(
        abs_v1=abs_v1, abs_v2=abs_v2, abs_dv=abs_dv,
        p_arr=p_arr, T_arr=T_arr, thickness_arr=thickness_lay,
        air_arr=air_lay, xO2_arr=xO2_arr, xN2_arr=xN2_arr, xH2O_arr=xH2O_arr,
        RHOAVE_arr=RHOAVE_arr, XKT_arr=XKT_arr, amagat_arr=amagat_lay,
        JRAD=JRAD
    )
    
    C_h2o_continuum = water_continuum_profile(
        abs_v1=abs_v1, abs_v2=abs_v2, abs_dv=abs_dv,
        p_arr=p_arr, T_arr=T_arr, thickness_arr=thickness_lay,
        air_arr=air_lay, xH2O_arr=xH2O_arr,
        JRAD=JRAD
    )
    


    # Add all contributions to the total absorption profile
    
    
    ABSRB_profile += C_co2_all
    ABSRB_profile_tmp = ABSRB_profile.copy()
    ABSRB_profile += C_o3_all
    # save beyond 40800 to avoid double‐counting
    if o3_diff_V1C is not None:
        o3_VJ = o3_diff_V1C + o3_diff_DVC * (o3_diff_NPTO3-1)
        if abs_v2 > 27370.0 and abs_v1 < 40800.0 and XO3CN > 0.0:
            if o3_VJ > 40815.0 and abs_v2 > 40800.0:
                i_fix = int((40800.0 - abs_v1) / abs_dv + 1.001)
                ABSRB_profile[i_fix:] = ABSRB_profile_tmp[i_fix : npt_abs]
        elif abs_v2 > 40800.0 and abs_v1 < 54000.0 and XO3CN > 0.0:
            if abs_v1 < 40800.0:
                i_fix = int((40800.0 - abs_v1) / abs_dv + 1.001)
                ABSRB_profile[: i_fix] = ABSRB_profile_tmp[:i_fix]
    
    ABSRB_profile += C_o2_ver1_all
    ABSRB_profile += C_o2_collision_all
    ABSRB_profile += C_o2_continuum_all
    ABSRB_profile += C_n2_continuum
    ABSRB_profile += C_h2o_continuum



    return nu_list, ABSRB_profile

def g_distribution(wvl, tau, solar_flux, ind_sort, g_num=16, weight=None):
    weight_default = np.array([0,\
                    0.1527534276, 0.1491729617, 0.1420961469, 0.1316886544, \
                    0.1181945205, 0.1019300893, 0.0832767040, 0.0626720116, \
                    0.0424925000, 0.0046269894, 0.0038279891, 0.0030260086, \
                    0.0022199750, 0.0014140010, 0.0005330000, 0.000075])
    wvl_num = wvl.size
    if weight is None:
        g_num = 16
        weight = weight_default
        print('Using default weights and g=16 for g-distribution.')
    
    assert weight.size == g_num+1, 'weight size must be g_num'
    
    weight_cum = np.cumsum(weight)
    g = np.arange(wvl_num)/wvl_num
    
    tau_all_sorted = np.zeros_like(tau)
    
    ind_sort = np.argsort(tau[0, :])

    # plt.figure(figsize=(20, 9))
    # plt.plot(g, tau[0, ind_sort], label='Layer 0')
    # plt.xlabel('g')
    # plt.ylabel('Optical Depth')
    # plt.title('Optical Depth vs g for Layer 0')
    # plt.legend()
    # plt.show()
    # # Sort tau and solar_flux according to ind_sort
    # sys.exit()
    
    
    for iz in range(tau.shape[0]):
        tau_all_sorted[iz, :] = tau[iz, ind_sort]
    solar_flux_sorted = solar_flux[ind_sort]
    solar_flux_sorted /= 1000  # Convert to W/m^2/nm
    
    plt.figure(figsize=(20, 9))
    for iz in range(tau.shape[0]):
        plt.plot(g, tau_all_sorted[iz, :], label=iz)
    ymin, ymax = plt.ylim()
    plt.vlines(weight_cum, ymin, ymax, linestyle='--', color='grey', )
    plt.yscale('log')
    plt.legend()
    plt.show()
    
    tau_g_total = np.zeros((tau.shape[0], g_num))
    solar_g = np.zeros(g_num)
    for j in range(g_num):
        g_mask = np.logical_and(g>=weight_cum[j], g<weight_cum[j+1])
        print(np.mean(tau_all_sorted[:, g_mask], axis=1).shape)
        tau_g_total[:, j] = np.mean(tau_all_sorted[:, g_mask], axis=1)
        solar_g[j] = np.mean(solar_flux_sorted[g_mask])
        
    return tau_g_total, solar_g, weight


if __name__ == '__main__':

    import pickle as pkl

    fname_atm = '/Users/yuch8913/programming/er3t/er3t_mca_v11/er3t/examples/tmp-data/00_er3t_mca/example_05_rad_les_cloud_3d/atm.pk'

    levels = np.arange(0.0, 40.1, 4)

    atm0   = er3t.pre.atm.atm_atmmod(levels=levels, fname=fname_atm, fname_atmmod=er3t.common.params['atmospheric_profile'], overwrite=False)

    wv1abs = 762.0
    wv2abs = 778.0
    slit_wvl = np.arange(wv1abs, wv2abs+1e-7, 0.1)
    slit_response = np.ones_like(slit_wvl)
    
    ssfr_slit_vis_file = "/Users/yuch8913/programming/er3t/er3t/er3t/data/slit/ssfr/vis_0.1nm_s.dat"
    slit_data = read_dat(ssfr_slit_vis_file)
    slit_response = np.interp(slit_wvl-770, slit_data[:, 0], slit_data[:, 1])
    
    if 1:#not os.path.exists('tmp_abs.pkl'):
        nu_final, cont_tau_final, lbl_tau_final, solar_data_interpolate, slit_response_final, ind_sort = abs_tau_calc(atm0=atm0,
                                wv1abs=wv1abs, wv2abs=wv2abs, dvabs=0.01, 
                                slit_wvl=slit_wvl, slit_response=slit_response,
                                radflag=True)
        
        # save nu, cont_tau_final, lbl_tau_final, solar_data_interpolate, slit_response_final, ind_sort
        with open('tmp_abs.pkl', 'wb') as f:
            pkl.dump((nu_final, cont_tau_final, lbl_tau_final, solar_data_interpolate, slit_response_final, ind_sort), f)
    else:
        with open('tmp_abs.pkl', 'rb') as f:
            nu_final, cont_tau_final, lbl_tau_final, solar_data_interpolate, slit_response_final, ind_sort = pkl.load(f)
    
    
    # print("cont_tau_final iz=0:", cont_tau_final[0, :])
    # print("lbl_tau_final iz=0:", lbl_tau_final[0, :])
    # print(1e7 / nu_final)
    # sys.exit()
    final_tau = (cont_tau_final + lbl_tau_final) * slit_response_final
    tau_g_total, solar_g, weight = g_distribution(1e7/nu_final, final_tau, solar_data_interpolate, ind_sort, g_num=16)
    
    # for iz in range(tau_g_total.shape[0]):
    #     print(f"Layer {iz}: ")
    #     print("tau_g:", tau_g_total[iz, :])
    # sys.exit()

    wvl_nm = 1e7 / nu_final  # Convert wavenumber to wavelength in nm
    print('max abs:', cont_tau_final.max(), cont_tau_final.max())
    for iz in range(cont_tau_final.shape[0]):
        plt.plot(wvl_nm, cont_tau_final[iz, :], label='z={:01d}'.format(iz))
    # plt.plot(wvl_nm, ds[1], label='Foreign Absorption')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Absorption optical depth')
    plt.legend()
    plt.yscale('log')
    plt.show()
    
    for iz in range(lbl_tau_final.shape[0]):
        plt.plot(wvl_nm, lbl_tau_final[iz, :], label='z={:01d}'.format(iz))
    # plt.plot(wvl_nm, ds[1], label='Foreign Absorption')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('LBL Absorption optical depth')
    plt.legend()
    plt.yscale('log')
    plt.show()
    
    for iz in range(1):
        plt.plot(wvl_nm, final_tau[iz, :], label='z={:01d}'.format(iz))
    # plt.plot(wvl_nm, ds[1], label='Foreign Absorption')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Absorption optical depth at 1st layer')
    plt.legend()
    plt.yscale('log')
    plt.show()
    
    for iz in range(lbl_tau_final.shape[0]):
        plt.plot(wvl_nm, final_tau[iz, :], label='z={:01d}'.format(iz))
    # plt.plot(wvl_nm, ds[1], label='Foreign Absorption')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Absorption optical depth at each layer')
    plt.legend()
    plt.yscale('log')
    plt.show()
    

    plt.plot(wvl_nm, np.sum(lbl_tau_final[:, :]+cont_tau_final[:, :], axis=0))
    # plt.plot(wvl_nm, ds[1], label='Foreign Absorption')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Sum Absorption optical depth for all layers')
    plt.legend()
    plt.yscale('log')
    plt.show()

    pass
