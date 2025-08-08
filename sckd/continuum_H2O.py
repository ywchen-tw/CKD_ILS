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
import numpy as np
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


continuum_data_dir = os.path.join(os.path.dirname(__file__), 'data')

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

def myradfn_h2o(vi, xkt, T):
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

def xint_h2o(v1a, v2a, dva, a, afact, vft, dvr3, nptabs, n1r3, n2r3):
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
    dat = read_xarray_file(f"{continuum_data_dir}/absco-ref_wv-mt-ckd.nc", FRGNX)
    version = dat['title']
    if mt_version is not None:
        mt_version = version[3:50]

    wvn = dat['wavenumber']
    dvc = wvn[1] - wvn[0]
    i = 0
    while i < len(wvn) and wvn[i] <= (nu1abs - 2 * dvc):
        i += 1
    i1 = max(i - 1, 0)
    while i < len(wvn) and wvn[i] < (nu2abs + 2 * dvc):
        i += 1
    i2 = i
    ncoeff = i2 - i1
    
    if ncoeff <= 1:
        nuout_arr = np.arange(nu1abs, nu2abs + 1e-5, dvabs)
        self_absco = np.zeros_like(nuout_arr)
        for_absco = np.zeros_like(nuout_arr)
        
        return self_absco, for_absco, mt_version, None, nuout_arr
    
    else:

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
            rad = myradfn_h2o(wvn[i1:i2], xkt, t_atm)
            print("max rad:", rad.max(), "min:", rad.min())
            # plt.plot(wvn[i1:i2], rad, label='Radiation Term')
            # plt.xlabel('Wavenumber (cm⁻¹)')
            # plt.ylabel('Radiation Correction Factor')
            # plt.show()
            sh2o_coeff *= rad
        else:
            rad = np.zeros_like(sh2o_coeff)
        # print("max sh2o_coeff after rad ajustment:", sh2o_coeff.max(), "min:", sh2o_coeff.min())

        # Interpolate coefficients to output spectral grid.
        # nptabs = int((nu2abs - nu1abs) / dvabs + 1)
        nuout_arr = np.arange(nu1abs, nu2abs + 1e-5, dvabs)
        # ist, lst = pre_xint_h2o(wvn[0], wvn[-1], nu1abs, dvabs, nptabs)
        # self_absco = xint_h2o(wvn[i1], wvn[i2 - 1], dvc, sh2o_coeff, 1.0, nu1abs, dvabs, nptabs, ist, lst)

        VC_grid = wvn[i1:i2]
        abs_grid = np.arange(nu1abs, nu2abs + 1e-5, dvabs)
        self_absco = np.interp(abs_grid, VC_grid, sh2o_coeff)
        
        # *****************
        # Compute water vapor foreign continuum absorption coefficient.
        fh2o_coeff = dat['for_absco_ref'][i1:i2] * (1 - h2o_vmr) * rho_rat
        # print("max fh2o_coeff:", fh2o_coeff.max(), "min:", fh2o_coeff.min())
        
        # Multiply by radiation term if requested
        if radflag:
            fh2o_coeff *= rad

        # Interpolate coefficients to output spectral grid.
        # for_absco = xint_h2o(wvn[i1], wvn[i2 - 1], dvc, fh2o_coeff, 1.0, nu1abs, dvabs, nptabs, ist, lst)
        
        VC_grid = wvn[i1:i2]
        abs_grid = np.arange(nu1abs, nu2abs + 1e-5, dvabs)
        for_absco = np.interp(abs_grid, VC_grid, fh2o_coeff)
        
        
        # *****************
        
        return self_absco, for_absco, mt_version, wvn[i1:i2], nuout_arr


def compute_h2o_continuum(abs_v1, abs_v2, abs_dv,
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
    



if __name__ == '__main__':

    pass
