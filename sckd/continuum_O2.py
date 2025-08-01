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
    
    bo2_vis_data = pd.read_csv(f"{continuum_data_dir}/bo2in_vis.csv")
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
    bo2in_fuv_data = pd.read_csv(f"{continuum_data_dir}/bo2in_fuv.csv", header=None)
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


def compute_o2_continuum(abs_v1, abs_v2, abs_dv,
                   p_arr, T_arr, thickness_arr,
                   air_arr,
                   xO2_arr, xN2_arr, xH2O_arr,
                   RHOAVE_arr, XKT_arr, amagat_arr,
                   JRAD=False
                   ):
    """Compute the total O2 continuum absorption for a full vertical profile.
    This function combines the various O2 continuum absorption models
    into a single profile for the specified spectral range."""
    
    C_o2_ver1_all = o2_ver1_profile(
        abs_v1=abs_v1, abs_v2=abs_v2, abs_dv=abs_dv,
        p_arr=p_arr, T_arr=T_arr, thickness_arr=thickness_arr,
        air_arr=air_arr, xO2_arr=xO2_arr, 
        RHOAVE_arr=RHOAVE_arr,
        XKT_arr=XKT_arr,
        amagat_arr=amagat_arr,
        JRAD=JRAD
    )
    
    C_o2_collision_all = o2_collision_profile(
        abs_v1=abs_v1, abs_v2=abs_v2, abs_dv=abs_dv,
        p_arr=p_arr, T_arr=T_arr, thickness_arr=thickness_arr,
        air_arr=air_arr,
        xO2_arr=xO2_arr, xN2_arr=xN2_arr, xH2O_arr=xH2O_arr,
        RHOAVE_arr=RHOAVE_arr, XKT_arr=XKT_arr, amagat_arr=amagat_arr,
        JRAD=JRAD
    )
    
    C_o2_continuum_all = o2_continuum_profile(
        abs_v1=abs_v1, abs_v2=abs_v2, abs_dv=abs_dv,
        p_arr=p_arr, T_arr=T_arr, thickness_arr=thickness_arr,
        air_arr=air_arr, xO2_arr=xO2_arr,
        RHOAVE_arr=RHOAVE_arr, XKT_arr=XKT_arr, amagat_arr=amagat_arr,
        JRAD=JRAD
    )
    
    return C_o2_ver1_all + C_o2_collision_all + C_o2_continuum_all

if __name__ == '__main__':

    pass
