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
from .hapi import *
from .utils import linear_interp, read_dat
from .continuum_gas_compute import compute_continuum_profile
from .lbl_gas_compute import compute_lbl_profile



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


def abs_tau_calc(wv1abs, wv2abs, dvabs,
                 p_lay, t_lay, thickness_lay, air_lay,
                 x_vmr_co2, x_vmr_n2, x_vmr_o2, x_vmr_h2o, x_vmr_o3, x_vmr_ch4, x_vmr_no2, x_vmr_n2o,
                 slit_wvl, slit_response,
                 FRGNX=0, radflag=True, 
                 fname_solar="/Users/yuch8913/programming/er3t/er3t/er3t/data/solar/data/solar_flux/kurudz_full.dat"
                 ):
    # Constants equivalent to the Fortran DATA statements
    P0 = 1013.0
    T0 = 296.0
    XLOSMT = 2.68675e19
    
    icflg = 0
    
    
    if isinstance(p_lay, float):
        p_lay = np.array([p_lay])
        output = 'single layer'
    elif isinstance(p_lay, list) or isinstance(p_lay, np.ndarray):
        p_lay = np.array(p_lay)
        output = 'multiple layers'
    if isinstance(t_lay, float):
        t_lay = np.array([t_lay])
    if isinstance(thickness_lay, float):
        thickness_lay = np.array([thickness_lay])
    if isinstance(air_lay, float):
        air_lay = np.array([air_lay])
    if isinstance(x_vmr_co2, float):
        x_vmr_co2 = np.array([x_vmr_co2])
    if isinstance(x_vmr_n2, float):
        x_vmr_n2 = np.array([x_vmr_n2])
    if isinstance(x_vmr_o2, float):
        x_vmr_o2 = np.array([x_vmr_o2])
    if isinstance(x_vmr_h2o, float):
        x_vmr_h2o = np.array([x_vmr_h2o])
    if isinstance(x_vmr_o3, float):
        x_vmr_o3 = np.array([x_vmr_o3])
    if isinstance(x_vmr_ch4, float):
        x_vmr_ch4 = np.array([x_vmr_ch4])
    if isinstance(x_vmr_no2, float):
        x_vmr_no2 = np.array([x_vmr_no2])
    if isinstance(x_vmr_n2o, float):
        x_vmr_n2o = np.array([x_vmr_n2o])
    

    
    RHOAVE_lay = (p_lay/P0)*(T0/t_lay)
    XKT_lay = t_lay/RADCN2
    amagat_lay = (p_lay/P0)*(273./t_lay)
    
    nu1abs = 1e7 / wv2abs
    nu2abs = 1e7 / wv1abs
    nptabs = int((nu2abs - nu1abs) / dvabs + 1)
    nuout_arr = np.arange(nu1abs, nu2abs + 1e-5, dvabs)


    
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
    
    nlay = p_lay.shape[0]
    
    cont_tau_final_inter = np.zeros((nlay, nuout_arr.shape[0]))
    for iz in range(nlay):
        cont_tau_final_inter[iz, :] = linear_interp(nu_final, cont_tau_final[iz, :], nuout_arr)
    
    # print("lambda_final shape:", lambda_final.shape)

    nu_total, lambda_total, lbl_tau_final, \
    coef_h2o_final, coef_co2_final, coef_o3_final,\
    coef_ch4_final, coef_o2_final, coef_no2_total, coef_no2_total = compute_lbl_profile(nu1abs, nu2abs, dvabs,
                                                        p_lay, t_lay, thickness_lay, air_lay, 
                                                        x_vmr_co2, x_vmr_o2, x_vmr_h2o, x_vmr_o3, x_vmr_ch4, x_vmr_no2, x_vmr_n2o)
    
    
    # print("lambda_total shape:", lambda_total.shape)
    
    if not os.path.exists(fname_solar):
        raise FileNotFoundError(f"Solar data file not found: {fname_solar}")
    datContent = [i.strip().split() for i in open(fname_solar).readlines()]
    solar_data = np.array(datContent[11:]).astype(np.float32)
    
    solar_lambda_mask = np.logical_and(solar_data[:, 0]>=wv1abs, solar_data[:, 0]<=wv2abs)
    solar_data_interpolate = linear_interp(1.0e7/solar_data[:, 0], solar_data[:, 1], nu_final)
    print("solar_data_interpolate shape:", solar_data_interpolate.shape)
    
    slit_response_fit = interp1d(1.0e7/slit_wvl, slit_response, kind='linear', bounds_error=False, fill_value=0.0)
    slit_response_final = slit_response_fit(nu_final)
    
    tau_final = cont_tau_final_inter + lbl_tau_final
    
    for iz in range(nlay):
        print("layer:", iz)
        print("  cont_tau_final_inter max, min:", cont_tau_final_inter[iz, :].max(), cont_tau_final_inter[iz, :].min())
        print("  lbl_tau_final max, min:", lbl_tau_final[iz, :].max(), lbl_tau_final[iz, :].min())
        print("  tau_final max, min:", tau_final[iz, :].max(), tau_final[iz, :].min())
        
    
    # use surface layer
    ind_sort = np.argsort(tau_final[0, :]*slit_response_final)

    if output == 'single layer':
        cont_tau_final_inter = cont_tau_final_inter[0, :]
        lbl_tau_final = lbl_tau_final[0, :]
        

    return nu_final, cont_tau_final_inter, lbl_tau_final, solar_data_interpolate, slit_response_final, ind_sort

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
    
    
    
    if tau.ndim == 1:
        tau = tau[np.newaxis, :]
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
    
    # plt.figure(figsize=(20, 9))
    # for iz in range(tau.shape[0]):
    #     plt.plot(g, tau_all_sorted[iz, :], label=iz)
    # ymin, ymax = plt.ylim()
    # plt.vlines(weight_cum, ymin, ymax, linestyle='--', color='grey', )
    # plt.yscale('log')
    # plt.legend()
    # plt.show()
    
    tau_g_total = np.zeros((tau.shape[0], g_num))
    solar_g = np.zeros(g_num)
    for j in range(g_num):
        g_mask = np.logical_and(g>=weight_cum[j], g<weight_cum[j+1])
        print(np.mean(tau_all_sorted[:, g_mask], axis=1).shape)
        tau_g_total[:, j] = np.mean(tau_all_sorted[:, g_mask], axis=1)
        solar_g[j] = np.mean(solar_flux_sorted[g_mask])
        
    return tau_g_total, solar_g, weight


def compute_reference_tau_and_layer_ordinates(
        wavelengths, tau_highres, delta_m, solar_flux, N=8, alpha=2.0, m=3):
    """
    Compute reference (mean-layer) correlated-tau ordinates using a single CDF
    and return tau_ordinates for each layer. Supports 1D tau_highres (one layer)
    or 2D tau_highres (multiple layers).

    Parameters:
        wavelengths (1D array): High-res wavelength grid (µm).
        tau_highres (1D or 2D array): Optical depths τ(λ) or τ(λ, layer).
        delta_m (float or 1D array): Path mass for each layer.
        solar_flux (1D array): Solar spectral flux at each wavelength.
        N (int): Number of g-ordinates.
        alpha (float): Exponent for alternate mapping φ(g) = g^α/(g^α+(1−g)^α).

    Returns:
        tau_ordinates_layers (2D array): τ-values per layer (n_layers, N).
        weights_ref (N-array): Fixed quadrature weights.
        flux_ordinates (N-array): Solar flux corresponding to each g-ordinate.
    """
    # Ensure tau_highres is 2D array (n_wav, n_layers)
    tau_arr = np.atleast_2d(tau_highres)
    print("tau_highres shape:", tau_arr.shape)
    print("tau_arr shape:", tau_arr.shape)
    if tau_arr.shape[1] == len(wavelengths):
        tau_arr = tau_arr.T
    n_wav, n_layers = tau_arr.shape

    # Ensure delta_m is 1D array of length n_layers
    delta_arr = np.atleast_1d(delta_m)
    if delta_arr.size == 1:
        delta_arr = np.full(n_layers, delta_arr)

    # 1. Compute reference optical depth: sum over layers (no solar weighting)
    tau_ref = np.tensordot(tau_arr, delta_arr, axes=(1,0))  # shape (n_wav,)

    # 2. Build reference CDF
    sorted_idx = np.argsort(tau_ref)
    tau_ref_sorted = tau_ref[sorted_idx]
    flux_sorted = solar_flux[sorted_idx]
    g_space = np.arange(len(tau_ref_sorted))/len(tau_ref_sorted) # normalize to [0,1]

    # 3. Gaussian quadrature nodes & weights on [0,1]
    x, w = np.polynomial.legendre.leggauss(N)
    x_mapped = 0.5 * (x + 1)
    w_mapped = 0.5 * w

    # 4. Alternate mapping φ and φ'
    def phi(g):
        return (g**alpha) / (g**alpha + (1 - g)**alpha)

    def phi_prime(g):
        ga = g**alpha
        one_ga = (1 - g)**alpha
        num = alpha * g**(alpha - 1) * (ga + one_ga) - ga * alpha * (g**(alpha - 1) - (1 - g)**(alpha - 1))
        den = (ga + one_ga)**2
        return num / den

    # 5. Remap nodes & adjust weights
    g_mapped = phi(x_mapped)
    weights_ref = w_mapped * phi_prime(x_mapped)
    weights_ref /= np.sum(weights_ref)

    # 6. Interpolate per-layer τ into g-space
    
    # m-point Simpson’s rule per bin
    m = m  # Simpson needs odd number of points
    # Precompute relative positions and Simpson weights on [0,1]
    Epsilon = np.linspace(0, 1, m)                             # [0, 0.5, 1] for m=3
    if m == 3:
        W_simpson = np.array([1, 4, 1]) / (6/m)               # Simpson 1/3 weights sum to 1
    elif m == 4:
        W_simpson = np.array([1, 3, 3, 1]) / (8/m)            # Simpson 3/8 weights sum to 1 for m=4
    elif m == 5:
        W_simpson = np.array([1, 4, 6, 4, 1]) / (16/m)
    elif m == 6:
        W_simpson = np.array([1, 5, 10, 10, 5, 1]) / (32/m)
    elif m == 7:
        W_simpson = np.array([1, 6, 15, 20, 15, 6, 1]) / (64/m)
    else:
        raise ValueError(f"Unsupported m={m} for Simpson's rule. Use m=3, 4, 5, 6, or 7.")
    Δg = np.empty(N+1)
    # we need the bin edges g_edges; assuming g_mapped sorted ascending:
    g_edges = np.zeros(N+1)
    g_edges[1:-1] = 0.5*(g_mapped[:-1] + g_mapped[1:])    # midpoints between nodes
    g_edges[ 0]   = 0.0
    g_edges[-1]   = 1.0
    
    tau_ordinates_layers = np.zeros((n_layers, N))
    for l in range(n_layers):
        # tau_layer = tau_arr[:, l] * delta_arr[l]
        tau_layer = tau_arr[:, l]# * delta_arr[l]
        tau_sorted = tau_layer[sorted_idx]
        tau_ordinates_layers[l, :] = np.interp(g_mapped, g_space, tau_sorted)
        
        # plt.plot(g_space, tau_sorted, label=f'Layer {l+1}')
        # plt.plot(g_space, tau_ref[sorted_idx], 'k--', label='Reference τ')
        # plt.scatter(g_mapped, tau_ordinates_layers[l, :], label=f'g-ordinates Layer {l+1}')
        # plt.xlabel('g')
        # plt.ylabel('Optical Depth')
        # plt.yscale('log')
        # plt.show()
        
        # allocate new array
        tau_bin_means = np.zeros(N)
        for i in range(N):
            g0, g1 = g_edges[i], g_edges[i+1]
            # map Simpson nodes into this bin
            g_sub = g0 + Epsilon*(g1-g0)
            # interpolate tau at these subpoints
            tau_sub = np.interp(g_sub, g_space, tau_sorted)
            # Simpson‐weighted mean
            tau_bin_means[i] = np.dot(W_simpson, tau_sub)

        tau_ordinates_layers[l, :] = tau_bin_means

    # 7. Interpolate solar_flux into g-space (independent)
    # flux_ordinates = np.interp(g_mapped, g_space, flux_sorted)
    flux_ordinates = np.zeros(N)
    
    for i in range(N):
        g0, g1 = g_edges[i], g_edges[i+1]
        flux_ordinates[i] = np.mean(flux_sorted[np.logical_and(g_space >= g0, g_space < g1)])
    
    # for i in range(N):
    #     g0, g1 = g_edges[i], g_edges[i+1]
    #     # map Simpson nodes into this bin
    #     g_sub = g0 + Epsilon*(g1-g0)
    #     # interpolate solar flux at these subpoints
    #     flux_sub = np.interp(g_sub, g_space, flux_sorted)
    #     # Simpson‐weighted mean
    #     flux_bin_means[i] = np.dot(W_simpson, flux_sub)
    # flux_ordinates = flux_bin_means
    
    if n_layers == 1:
        tau_ordinates_layers = tau_ordinates_layers.flatten()

    return tau_ordinates_layers, flux_ordinates, weights_ref,


def compute_ckd_tau_and_layer_ordinates(
        wavelengths, tau_highres, solar_flux, N=8, alpha=2.0, m=3, layer_indices=0,):
    """
    Compute reference (mean-layer) correlated-tau ordinates using a single CDF
    and return tau_ordinates for each layer. Supports 1D tau_highres (one layer)
    or 2D tau_highres (multiple layers).

    Parameters:
        wavelengths (1D array): High-res wavelength grid (µm).
        tau_highres (1D or 2D array): Optical depths τ(λ) or τ(λ, layer).
        delta_m (float or 1D array): Path mass for each layer.
        solar_flux (1D array): Solar spectral flux at each wavelength.
        N (int): Number of g-ordinates.
        alpha (float): Exponent for alternate mapping φ(g) = g^α/(g^α+(1−g)^α).
        layer_indices (list or array): Indices of layers to compute ordinates for.

    Returns:
        tau_ordinates_layers (2D array): τ-values per layer (n_layers, N).
        weights_ref (N-array): Fixed quadrature weights.
        flux_ordinates (N-array): Solar flux corresponding to each g-ordinate.
    """
    # Ensure tau_highres is 2D array (n_wav, n_layers)
    tau_arr = np.atleast_2d(tau_highres)
    print("tau_highres shape:", tau_arr.shape)
    print("tau_arr shape:", tau_arr.shape)
    if tau_arr.shape[1] == len(wavelengths):
        tau_arr = tau_arr.T
    n_wav, n_layers = tau_arr.shape

    if layer_indices > n_layers:
        raise ValueError(f"layer_indices {layer_indices} exceeds number of layers {n_layers}")
    # 1. Compute reference optical depth: sum over layers (no solar weighting)
    tau_ref = tau_arr[:, layer_indices]

    # 2. Build reference CDF
    sorted_idx = np.argsort(tau_ref)
    tau_ref_sorted = tau_ref[sorted_idx]
    flux_sorted = solar_flux[sorted_idx]
    g_space = np.arange(len(tau_ref_sorted))/len(tau_ref_sorted) # normalize to [0,1]

    # 3. Gaussian quadrature nodes & weights on [0,1]
    x, w = np.polynomial.legendre.leggauss(N)
    x_mapped = 0.5 * (x + 1)
    w_mapped = 0.5 * w

    # 4. Alternate mapping φ and φ'
    def phi(g):
        return (g**alpha) / (g**alpha + (1 - g)**alpha)

    def phi_prime(g):
        ga = g**alpha
        one_ga = (1 - g)**alpha
        num = alpha * g**(alpha - 1) * (ga + one_ga) - ga * alpha * (g**(alpha - 1) - (1 - g)**(alpha - 1))
        den = (ga + one_ga)**2
        return num / den

    # 5. Remap nodes & adjust weights
    g_mapped = phi(x_mapped)
    weights_ref = w_mapped * phi_prime(x_mapped)
    weights_ref /= np.sum(weights_ref)

    # 6. Interpolate per-layer τ into g-space
    
    # m-point Simpson’s rule per bin
    m = m  # Simpson needs odd number of points
    # Precompute relative positions and Simpson weights on [0,1]
    Epsilon = np.linspace(0, 1, m)                             # [0, 0.5, 1] for m=3
    if m == 3:
        W_simpson = np.array([1, 4, 1]) / (6/m)               # Simpson 1/3 weights sum to 1
    elif m == 4:
        W_simpson = np.array([1, 3, 3, 1]) / (8/m)            # Simpson 3/8 weights sum to 1 for m=4
    elif m == 5:
        W_simpson = np.array([1, 4, 6, 4, 1]) / (16/m)
    elif m == 6:
        W_simpson = np.array([1, 5, 10, 10, 5, 1]) / (32/m)
    elif m == 7:
        W_simpson = np.array([1, 6, 15, 20, 15, 6, 1]) / (64/m)
    else:
        raise ValueError(f"Unsupported m={m} for Simpson's rule. Use m=3, 4, 5, 6, or 7.")
    Δg = np.empty(N+1)
    # we need the bin edges g_edges; assuming g_mapped sorted ascending:
    g_edges = np.zeros(N+1)
    g_edges[1:-1] = 0.5*(g_mapped[:-1] + g_mapped[1:])    # midpoints between nodes
    g_edges[ 0]   = 0.0
    g_edges[-1]   = 1.0
    
    
    tau_ordinates_layers = np.zeros((n_layers, N))
    for l in range(n_layers):
        # tau_layer = tau_arr[:, l] * delta_arr[l]
        tau_layer = tau_arr[:, l]# * delta_arr[l]
        tau_sorted = tau_layer[sorted_idx]
        tau_ordinates_layers[l, :] = np.interp(g_mapped, g_space, tau_sorted)
        
        # plt.plot(g_space, tau_sorted, label=f'Layer {l+1}')
        # plt.plot(g_space, tau_ref[sorted_idx], 'k--', label='Reference τ')
        # plt.scatter(g_mapped, tau_ordinates_layers[l, :], label=f'g-ordinates Layer {l+1}')
        # plt.xlabel('g')
        # plt.ylabel('Optical Depth')
        # plt.yscale('log')
        # plt.show()
        
        

        # allocate new array
        tau_bin_means = np.zeros(N)
        for i in range(N):
            g0, g1 = g_edges[i], g_edges[i+1]
            # map Simpson nodes into this bin
            g_sub = g0 + Epsilon*(g1-g0)
            # interpolate tau at these subpoints
            tau_sub = np.interp(g_sub, g_space, tau_sorted)
            # Simpson‐weighted mean
            tau_bin_means[i] = np.dot(W_simpson, tau_sub)

        tau_ordinates_layers[l, :] = tau_bin_means

    # 7. Interpolate solar_flux into g-space (independent)
    # flux_ordinates = np.interp(g_mapped, g_space, flux_sorted)
    flux_ordinates = np.zeros(N)
    
    for i in range(N):
        g0, g1 = g_edges[i], g_edges[i+1]
        flux_ordinates[i] = np.mean(flux_sorted[np.logical_and(g_space >= g0, g_space < g1)])
    
    # for i in range(N):
    #     g0, g1 = g_edges[i], g_edges[i+1]
    #     # map Simpson nodes into this bin
    #     g_sub = g0 + Epsilon*(g1-g0)
    #     # interpolate solar flux at these subpoints
    #     flux_sub = np.interp(g_sub, g_space, flux_sorted)
    #     # Simpson‐weighted mean
    #     flux_bin_means[i] = np.dot(W_simpson, flux_sub)
    # flux_ordinates = flux_bin_means
    
    if n_layers == 1:
        tau_ordinates_layers = tau_ordinates_layers.flatten()

    return tau_ordinates_layers, flux_ordinates, weights_ref,

if __name__ == '__main__':

    pass
