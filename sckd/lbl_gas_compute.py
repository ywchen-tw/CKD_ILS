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
import pandas as pd
from .hapi import *
from .utils import linear_interp


continuum_data_dir = os.path.join(os.path.dirname(__file__), 'data')
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






def compute_lbl_profile(wv1abs, wv2abs, abs_dv,
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
    
    

if __name__ == '__main__':

    pass
