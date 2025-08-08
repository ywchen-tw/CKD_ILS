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
    bo3ch_nu_list = pd.read_csv(f"{continuum_data_dir}/bo3ch.csv")["nu"].values
    bo3ch_x = pd.read_csv(f"{continuum_data_dir}/bo3ch.csv")["x"].values
    bo3ch_y = pd.read_csv(f"{continuum_data_dir}/bo3ch.csv")["y"].values
    bo3ch_z = pd.read_csv(f"{continuum_data_dir}/bo3ch.csv")["z"].values
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
    bo3hh0_data = pd.read_csv(f"{continuum_data_dir}/bo3hh0.csv")
    
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
    bo3hh1_data = pd.read_csv(f"{continuum_data_dir}/bo3hh1.csv")
    
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
    bo3hh2_data = pd.read_csv(f"{continuum_data_dir}/bo3hh2.csv")
    
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

def compute_o3_continuum(abs_v1, abs_v2, abs_dv,
                   p_arr, T_arr, thickness_arr,
                   air_arr, xO3_arr, 
                   RHOAVE_arr, XKT_arr,
                   JRAD=False):
    """
    Compute diffuse ozone for an entire vertical profile of layers.
    """
    nlev = p_arr.size
    npt_abs = int((abs_v2 - abs_v1) / abs_dv) + 1
    abs_v1_extend = abs_v1 - abs_dv * 5
    abs_v2_extend = abs_v2 + abs_dv * 5
    npt_abs_extend = int((abs_v2_extend - abs_v1_extend) / abs_dv) + 1
    C_o3_diff_all = np.zeros((nlev, npt_abs), dtype=float)


    for i in range(nlev):
        V1C, V2C, DVC, NPTO3, C_O3_diff = o3_diffuse(
            V1ABS=abs_v1_extend, V2ABS=abs_v2_extend, xo3cn=XO3CN,
            tave=T_arr[i], thickness=thickness_arr[i],
            air=air_arr[i], xo3=xO3_arr[i],
            XKT=XKT_arr[i], RHOAVE=RHOAVE_arr[i],
            JRAD=JRAD
                    )
        
        if C_O3_diff is not None:
            # i_lo, i_hi, frac = pre_xint(
            #     cont_v1=V1C, cont_v2=V2C, cont_dv=DVC, npt_cont=NPTO3,
            #     abs_v1=abs_v1, abs_v2=abs_v2, abs_dv=abs_dv, npt_abs=npt_abs
            # )
            # abs_array = np.zeros(npt_abs, dtype=float)
            # xint(cont_values=C_O3_diff, i_lo=i_lo, i_hi=i_hi, frac=frac, abs_array=abs_array)
            
            VC_grid = np.linspace(V1C, V2C, NPTO3)
            abs_grid = np.linspace(abs_v1, abs_v2, npt_abs)
            abs_array = np.interp(abs_grid, VC_grid, C_O3_diff)

        else:
            abs_array = np.zeros(npt_abs, dtype=float)
            
            
        C_o3_diff_all[i, :] = abs_array

    return C_o3_diff_all, V1C, V2C, DVC, NPTO3



if __name__ == '__main__':

    pass
