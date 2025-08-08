"""
by Yu-Wen Chen (Yu-Wen.Chen@colorado.edu)

This code is a Python adaptation of the MT_CKD water vapor continuum model
originally written in Fortran.

Modified from:
  - Code repository: https://https://github.com/AER-RC/LBLRTM/blob/master/src/contnm.f90 (v12.17 release); accessed on 2025-5-29

"""

import os
import sys
import h5py
import numpy as np
import pandas as pd
from .utils import linear_interp, radfn_cal, pre_xint, xint


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
    bfco2 = pd.read_csv(f"{continuum_data_dir}/bfco2.csv")
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

def compute_co2_continuum(abs_v1, abs_v2, abs_dv,
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

            # i_lo, i_hi, frac = pre_xint(
            #     cont_v1=V1C, cont_v2=V2C, cont_dv=DVC, npt_cont=NPTC,
            #     abs_v1=abs_v1, abs_v2=abs_v2, abs_dv=abs_dv, npt_abs=npt_abs
            # )
            # abs_array = np.zeros(npt_abs, dtype=float)
            # xint(cont_values=C_co2, i_lo=i_lo, i_hi=i_hi, frac=frac, abs_array=abs_array)
            
            VC_grid = np.linspace(V1C, V2C, NPTC)
            abs_grid = np.linspace(abs_v1, abs_v2, npt_abs)
            abs_array = np.interp(abs_grid, VC_grid, C_co2)

        C_co2_all[i, :] = abs_array

    return C_co2_all


if __name__ == '__main__':

    pass
