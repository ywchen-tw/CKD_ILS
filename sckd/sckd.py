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
    
    nu_total, lambda_total, lbl_tau_final = compute_lbl_profile(wv1abs, wv2abs, dvabs, atm0)
    
    
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

    pass
