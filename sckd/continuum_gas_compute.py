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

from .utils import linear_interp, radfn_cal, pre_xint, xint
from .continuum_H2O import compute_h2o_continuum
from .continuum_O3 import compute_o3_continuum
from .continuum_N2 import compute_n2_continuum
from .continuum_CO2 import compute_co2_continuum
from .continuum_O2 import compute_o2_continuum


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
    C_co2_all = compute_co2_continuum(
        abs_v1=abs_v1, abs_v2=abs_v2, abs_dv=abs_dv,
        p_arr=p_arr, T_arr=T_arr, thickness_arr=thickness_lay,
        air_arr=air_lay, xCO2_arr=xCO2_arr,
        RHOAVE_arr=RHOAVE_arr,
        XKT_arr=XKT_arr,
        JRAD=JRAD
    )
    
    C_o3_all, o3_diff_V1C, o3_diff_V2C, o3_diff_DVC, o3_diff_NPTO3 = compute_o3_continuum(
        abs_v1=abs_v1, abs_v2=abs_v2, abs_dv=abs_dv,
        p_arr=p_arr, T_arr=T_arr, thickness_arr=thickness_lay,
        air_arr=air_lay, xO3_arr=xO3_arr, 
        RHOAVE_arr=RHOAVE_arr,
        XKT_arr=XKT_arr,
        JRAD=JRAD
    )
   
    # compute_o2_continuum includes both o2_ver1, o2_collision, and o2_continuum
    C_o2_continuum = compute_o2_continuum(
        abs_v1=abs_v1, abs_v2=abs_v2, abs_dv=abs_dv,
        p_arr=p_arr, T_arr=T_arr, thickness_arr=thickness_lay,
        air_arr=air_lay, xO2_arr=xO2_arr,
        xN2_arr=xN2_arr, xH2O_arr=xH2O_arr,
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
    
    C_h2o_continuum = compute_h2o_continuum(
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
    
    ABSRB_profile += C_o2_continuum
    ABSRB_profile += C_n2_continuum
    ABSRB_profile += C_h2o_continuum



    return nu_list, ABSRB_profile



if __name__ == '__main__':

    pass
