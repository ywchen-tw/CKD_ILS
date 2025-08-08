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
from .utils import linear_interp, radfn_cal, pre_xint, xint
# mpl.use('Agg')



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
            freq, c0, c1, V1C, V2C, DVC, NPTC_xn2_r = xn2_r(abs_v1, abs_v2, T_arr[i])
            C = tau_fac * c0 * (xN2_arr + c1*xO2_arr + 1*xH2O_arr)
            if JRAD == 1:
                C *= radfn_cal(freq, XKT_arr[i])
            
            # i_lo, i_hi, frac = pre_xint(
            #     cont_v1=V1C, cont_v2=V2C, cont_dv=DVC, npt_cont=NPTC_xn2_r,
            #     abs_v1=abs_v1, abs_v2=abs_v2, abs_dv=abs_dv, npt_abs=npt_abs
            # )
            # abs_array = np.zeros(npt_abs, dtype=float)
            # xint(cont_values=C, i_lo=i_lo, i_hi=i_hi, frac=frac, abs_array=abs_array)
            
            VC_grid = np.linspace(V1C, V2C, NPTC_xn2_r)
            abs_grid = np.linspace(abs_v1, abs_v2, npt_abs)
            abs_array = np.interp(abs_grid, VC_grid, C)
            
            C_n2_continuum_all[i, :] = abs_array

    # ── N2 fundamental
    if abs_v2 > 2001.77 and abs_v1 < 2897.59 and XN2CN > 0:
        for i in range(nlev):
            WKN2 = xN2_arr[i] * air_arr[i] # in molec/cm3
            WKN2_col = WKN2 * thickness_arr[i] * 1000 * 100 # in molec/cm2
            tau_fac = XN2CN * (WKN2_col/XLOSMT) * amagat_arr[i]
            freq, cn0, cn1, cn2, V1C, V2C, DVC, NPTC_N2_ver_1 = n2_ver_1(abs_v1, abs_v2, T_arr[i])
            C = tau_fac * (xN2_arr*cn0 + xO2_arr*cn1 + xH2O_arr*cn2)
            if JRAD == 1:
                C *= radfn_cal(freq, XKT_arr[i])
            
            # i_lo, i_hi, frac = pre_xint(
            #     cont_v1=V1C, cont_v2=V2C, cont_dv=DVC, npt_cont=NPTC_N2_ver_1,
            #     abs_v1=abs_v1, abs_v2=abs_v2, abs_dv=abs_dv, npt_abs=npt_abs
            # )
            # abs_array = np.zeros(npt_abs, dtype=float)
            # xint(cont_values=C, i_lo=i_lo, i_hi=i_hi, frac=frac, abs_array=abs_array)
            
            VC_grid = np.linspace(V1C, V2C, NPTC_N2_ver_1)
            abs_grid = np.linspace(abs_v1, abs_v2, npt_abs)
            abs_array = np.interp(abs_grid, VC_grid, C)
            
            C_n2_continuum_all[i, :] = abs_array

    # ── N2 first overtone
    if abs_v2 > 4340 and abs_v1 < 4910 and XN2CN > 0:
        for i in range(nlev):
            WKN2 = xN2_arr[i] * air_arr[i] # in molec/cm3
            WKN2_col = WKN2 * thickness_arr[i] * 1000 * 100 # in molec/cm2
            tau_fac = XN2CN * (WKN2_col/XLOSMT) * amagat_arr[i] * \
                    (xN2_arr[i] + 1*xO2_arr[i] + 1*xH2O_arr[i])
            freq, C0, V1C, V2C, DVC, NPTC_N2_overtone1 = n2_overtone1(abs_v1, abs_v2)
            C = tau_fac * C0
            if JRAD == 1:
                C *= radfn_cal(freq, XKT_arr[i])
                
            # i_lo, i_hi, frac = pre_xint(
            #     cont_v1=V1C, cont_v2=V2C, cont_dv=DVC, npt_cont=NPTC_N2_overtone1,
            #     abs_v1=abs_v1, abs_v2=abs_v2, abs_dv=abs_dv, npt_abs=npt_abs
            # )
            # abs_array = np.zeros(npt_abs, dtype=float)
            # xint(cont_values=C, i_lo=i_lo, i_hi=i_hi, frac=frac, abs_array=abs_array)
            
            VC_grid = np.linspace(V1C, V2C, NPTC_N2_overtone1)
            abs_grid = np.linspace(abs_v1, abs_v2, npt_abs)
            abs_array = np.interp(abs_grid, VC_grid, C)
            
            C_n2_continuum_all[i, :] = abs_array

    return C_n2_continuum_all




if __name__ == '__main__':

    pass
