import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import h5py
import pandas as pd
import er3t
import datetime as dt
from importmonkey import add_path
add_path("/Users/yuch8913/programming/CKD_ILS/")
import sckd

def test_er3t_atm(modis='terra', band=1, Ng=8, alpha=1.5, m=3):
    
    
    
    fname_atm = '/Users/yuch8913/programming/er3t/er3t_mca_v11/er3t/examples/tmp-data/00_er3t_mca/example_05_rad_les_cloud_3d/atm.pk'
    fname_atm = '/Users/yuch8913/programming/les/test_ARCSIX_modis_dropsonde/output/arcsix_modis_dropsonde_atm_example.pk'

    levels = np.arange(0.0, 40.1, 2)
    # levels = np.concatenate((np.arange(0, 2.1, 0.2), 
    #                         np.arange(2.5, 4.1, 0.5), 
    #                         np.arange(5.0, 10.1, 2.5),
    #                         np.array([15, 20, 30., 40., 50.])))

    atm0   = er3t.pre.atm.atm_atmmod(levels=levels, fname=fname_atm, fname_atmmod=er3t.common.params['atmospheric_profile'], overwrite=False)
    
    zpt_filedir = './zpt_dir'
    os.makedirs(zpt_filedir, exist_ok=True)
    # write out the atmospheric profile in ascii format
    with open(f'{zpt_filedir}/atm_profiles_main_{modis}_{band}.dat', 'w') as f:
        header = ('# Adjusted MODIS 07 atmospheric profile\n'
                '#      z(km)      p(mb)        T(K)    air(cm-3)    o3(cm-3)     o2(cm-3)    h2o(cm-3)    co2(cm-3)     no2(cm-3)\n'
                )
        # Build all profile lines in one go.
        lines = [
                f'{atm0.lev["altitude"]["data"][i]:11.3f} {atm0.lev["pressure"]["data"][i]:11.5f} {atm0.lev["temperature"]["data"][i]:11.3f} '
                f'{atm0.lev["air"]["data"][i]:12.6e} {atm0.lev["o3"]["data"][i]:12.6e} {atm0.lev["o2"]["data"][i]:12.6e} '
                f'{atm0.lev["h2o"]["data"][i]:12.6e} {atm0.lev["co2"]["data"][i]:12.6e} {atm0.lev["no2"]["data"][i]:12.6e}'
                for i in range(len(atm0.lev['altitude']['data']))[::-1]
                ]
        f.write(header + "\n".join(lines))
    
    with open(f'{zpt_filedir}/atm_profiles_ch4_{modis}_{band}.dat', 'w') as f:  
        header = ('# Adjusted MODIS 07 atmospheric profile for ch4 only\n'
                '#      z(km)      ch4(cm-3)\n'
                )
        lines = [
                f'{atm0.lev["altitude"]["data"][i]:11.3f} {atm0.lev["ch4"]["data"][i]:12.6e}'
                for i in range(len(atm0.lev['altitude']['data']))[::-1]
                ]
        f.write(header + "\n".join(lines))
    # =================================================================================
    
    
    p_lay = atm0.lay['pressure']['data']
    t_lay = atm0.lay['temperature']['data']
    thickness_lay = atm0.lay['thickness']['data']
    
    #'air', 'o3', 'o2', 'h2o', 'co2', 'no2'
    h2o_lay = atm0.lay['h2o']['data']
    o3_lay = atm0.lay['o3']['data']
    o2_lay = atm0.lay['o2']['data']
    co2_lay = atm0.lay['co2']['data']
    no2_lay = atm0.lay['no2']['data']
    air_lay = atm0.lay['air']['data']
    n2_lay = air_lay - h2o_lay - o2_lay - co2_lay - no2_lay
    
    x_vmr_h2o = h2o_lay/air_lay
    x_vmr_o2  = o2_lay/air_lay
    x_vmr_co2 = co2_lay/air_lay
    x_vmr_no2 = no2_lay/air_lay
    x_vmr_ch4 = np.zeros_like(o2_lay)  # Assuming no CH4 in this case
    if 'ch4' in atm0.lay:
        x_vmr_ch4 = atm0.lay['ch4']['data']/air_lay
    else:
        x_vmr_ch4[...] = 0 # 1.93 ppm
    if 'n2o' in atm0.lay:
        x_vmr_n2o = atm0.lay['n2o']['data']/air_lay
    else:
        x_vmr_n2o = 0 # 330 ppb
    x_vmr_o3  = o3_lay/air_lay
    
    x_vmr_n2  = 1 - x_vmr_h2o - x_vmr_o2 - x_vmr_co2 - x_vmr_no2 - x_vmr_n2o - x_vmr_ch4 - x_vmr_o3
        

    if modis == 'terra':
        # read terra_modis_RSR.nc
        
        fname_modis = 'terra_modis_RSR.nc'
        
    with h5py.File(fname_modis, 'r') as f:
        band_wvl = f['bands'][:][band-1] # in nm
        RSR = f['RSR'][:][band-1, :]  # RSR for the band
        wvl_all = f['wavelength'][:]  # all wavelengths in nm
        
    RSR_mask = RSR > 1e-6
    wvl_select = wvl_all[RSR_mask]
    
    wv1abs = wvl_select.min()
    wv2abs = wvl_select.max()
    mean_wvl = band_wvl
    slit_wvl = wvl_select
    slit_response = RSR[RSR_mask]
    
    print("wv1abs:", wv1abs)
    print("wv2abs:", wv2abs)
    # sys.exit()

    if not os.path.exists(f'tmp_abs_er3t_test_{modis}_{band}.pkl'):
        nu_final, cont_tau_final, lbl_tau_final, \
        solar_data_interpolate, slit_response_final, ind_sort = sckd.abs_tau_calc(wv1abs=wv1abs, wv2abs=wv2abs, dvabs=0.01,
                                                                                    p_lay=p_lay, t_lay=t_lay, thickness_lay=thickness_lay, air_lay=air_lay,
                                                                                    x_vmr_co2=x_vmr_co2, 
                                                                                    x_vmr_n2=x_vmr_n2, 
                                                                                    x_vmr_o2=x_vmr_o2, 
                                                                                    x_vmr_h2o=x_vmr_h2o, 
                                                                                    x_vmr_o3=x_vmr_o3, 
                                                                                    x_vmr_ch4=x_vmr_ch4, 
                                                                                    x_vmr_no2=x_vmr_no2,
                                                                                    x_vmr_n2o=x_vmr_n2o,
                                                                                    slit_wvl=slit_wvl, slit_response=slit_response,
                                                                                    radflag=True,
                                                                                    fname_solar="/Users/yuch8913/programming/er3t/er3t/er3t/data/solar/data/solar_flux/kurudz_full.dat"
                                                                                    )
        
        # save nu, cont_tau_final, lbl_tau_final, solar_data_interpolate, slit_response_final, ind_sort
        with open(f'tmp_abs_er3t_test_{modis}_{band}.pkl', 'wb') as f:
            pkl.dump((nu_final, cont_tau_final, lbl_tau_final, solar_data_interpolate, slit_response_final, ind_sort), f)
    else:
        with open(f'tmp_abs_er3t_test_{modis}_{band}.pkl', 'rb') as f:
            nu_final, cont_tau_final, lbl_tau_final, solar_data_interpolate, slit_response_final, ind_sort = pkl.load(f)


    # print("cont_tau_final iz=0:", cont_tau_final[0, :])
    # print("lbl_tau_final iz=0:", lbl_tau_final[0, :])
    # print(1e7 / nu_final)
    # sys.exit()
    cont_tau_final_slit = cont_tau_final * slit_response_final
    lbl_tau_final_slit = lbl_tau_final * slit_response_final
    final_tau = cont_tau_final_slit + lbl_tau_final_slit
    tau_g_total, solar_g, weight = sckd.g_distribution(1e7/nu_final, final_tau, solar_data_interpolate, ind_sort, g_num=16)
    
    tau_g_total, solar_g, weight = sckd.compute_reference_tau_and_layer_ordinates(
                                        wavelengths=1e7/nu_final, 
                                        tau_highres=final_tau, 
                                        delta_m=1, solar_flux=solar_data_interpolate, N=Ng, alpha=alpha, m=m)
    

    
    tau_g_total_sfc, solar_g_sfc, weight_sfc = sckd.compute_ckd_tau_and_layer_ordinates(
                                        wavelengths=1e7/nu_final, 
                                        tau_highres=final_tau, 
                                        solar_flux=solar_data_interpolate, N=Ng, alpha=alpha, layer_indices=0, m=m)

    print("tau_g_total shape:", tau_g_total.shape)
    print("solar_g shape:", solar_g.shape)
    print("weight shape:", weight.shape)
    
    print("tau_g_total_sfc shape:", tau_g_total_sfc.shape)
    print("solar_g_sfc shape:", solar_g_sfc.shape)
    print("weight_sfc shape:", weight_sfc.shape)

    # save the results as a pickle file
    g_tau_solar_weight = {"levels": levels,
                          "tau_g_total": tau_g_total,
                          "solar_g": solar_g,
                          "weight": weight,}
    with open(f'g_tau_solar_weight_{modis}_{band}_{Ng}_{alpha}_{m}.pkl', 'wb') as f:
        pkl.dump(g_tau_solar_weight, f)
        
    g_tau_solar_weight_sfc = {"levels": levels,
                            "tau_g_total": tau_g_total_sfc,
                            "solar_g": solar_g_sfc,
                            "weight": weight_sfc,}
    with open(f'g_tau_solar_weight_sfc_{modis}_{band}_{Ng}_{alpha}_{m}.pkl', 'wb') as f:
        pkl.dump(g_tau_solar_weight_sfc, f)

    wvl_nm = 1e7 / nu_final  # Convert wavenumber to wavelength in nm
    print('max abs:', cont_tau_final.max(), cont_tau_final.max())
    
    
    # for iz in range(cont_tau_final.shape[0]):
    #     plt.plot(wvl_nm, cont_tau_final[iz, :], label='z={:01d}'.format(iz))
    # # plt.plot(wvl_nm, ds[1], label='Foreign Absorption')
    # plt.xlabel('Wavelength (nm)')
    # plt.ylabel('Absorption optical depth')
    # plt.legend()
    # plt.yscale('log')
    # plt.show()
    
    # plt.plot(wvl_nm, final_tau[0, :], 'k-', label='Total Absorption')
    # plt.plot(wvl_nm, cont_tau_final_slit[0, :], 'r-', label='Continuum Absorption', alpha=0.5)
    # plt.plot(wvl_nm, lbl_tau_final_slit[0, :], 'b-', label='LBL Absorption', alpha=0.5)
    # plt.legend()
    # # plt.plot(wvl_nm, ds[1], label='Foreign Absorption')
    # plt.xlabel('Wavelength (nm)')
    # plt.ylabel('Absorption optical depth')
    # plt.yscale('log')
    # plt.show()

def gaussian(x, mu, sig):
    return (
        1.0 / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.0) / 2)
    )


def lrt_test(lw=False, modis='terra', band=1, Ng=8, alpha=1.5, m=3):
    """
    Test the libRadtran simulation
    """

    with open(f'g_tau_solar_weight_{modis}_{band}_{Ng}_{alpha}_{m}.pkl', 'rb') as f:
        g_tau_solar_weight = pkl.load(f)
    levels = g_tau_solar_weight['levels']
    tau_g_total, solar_g, weight = g_tau_solar_weight['tau_g_total'], g_tau_solar_weight['solar_g'], g_tau_solar_weight['weight']
    g_number = len(weight)
    # xx_wvl_grid = np.arange(360, 1990.1, 5)
    with open(os.path.join('.', f'tau_g_total_{modis}_{band}_{Ng}.dat'), 'w') as f_grid:
        ### Build all profile lines in one go.
        # header = ('# levels ' + ' '.join(['%12d' % gi for gi in range(g_number)]) + '\n'
        #             )
        header = (' '.join(['%12.6e' % levels[i] for i in range(len(levels))[::-1]]) + '\n'
        )
        lines = [
                 f'{(gi+1.0)*100:12.2f} ' + ' '.join(['%12.6e' % tau_g_total[i, gi] for i in range(len(levels)-1)[::-1]])
                 for gi in range(g_number)
                ]
        f_grid.write(header + "\n".join(lines))
        # f_grid.write("\n".join(lines))
    
    with open('solar_g.dat', 'w') as f_solar:
            header = ('# SSFR version solar flux\n'
                    '# wavelength (nm)      flux (mW/m^2/nm)\n'
                    )
            # Build all profile lines in one go.
            lines = [
                    f'{(gi+1.0)*100:12.6f} '
                    f'{solar_g[gi]:12.6e}'
                    for gi in range(g_number)
                    ]
            f_solar.write(header + "\n".join(lines))
    

    atm_z_grid = levels
    z_list = atm_z_grid
    atm_z_grid_str = ' '.join(['%.2f' % z for z in atm_z_grid])
    # rt initialization
    #/----------------------------------------------------------------------------\#
    lrt_cfg = er3t.rtm.lrt.get_lrt_cfg()
    zpt_filedir = './zpt_dir'
    if not lw:
        lrt_cfg['atmosphere_file'] = f'{zpt_filedir}/atm_profiles_main_{modis}_{band}.dat'
        # lrt_cfg['atmosphere_file'] = lrt_cfg['atmosphere_file'].replace('afglus.dat', 'afglss.dat')
        # lrt_cfg['solar_file'] = None
        lrt_cfg['solar_file'] = None
        lrt_cfg['solar_file'] = 'solar_g.dat'
        lrt_cfg['number_of_streams'] = 16
        # lrt_cfg['mol_abs_param'] = 'reptran coarse'
        # lrt_cfg['mol_abs_param'] = f'reptran medium'
        lrt_cfg['output_process'] = 'per_band'
        input_dict_extra = {
                            # 'crs_model': 'rayleigh Bodhaine29',
                            # 'crs_model': 'rayleigh Nicolet',
                            # 'crs_model': 'o3 Bogumil',
                            # 'atm_z_grid': atm_z_grid_str,
                            'source': 'solar '+'solar_g.dat'+' per_band',
                            'mol_tau_file': 'abs ' + os.path.join('.', f'tau_g_total_{modis}_{band}_{Ng}.dat'),
                            # 'no_scattering':'mol',
                            # 'no_absorption':'mol',
                            # 'output_quantity': 'reflectivity',
                            # 'output_process': 'per_band',
                            }
        Nx_effective = g_number
        mute_list = ['solar_file', 'wavelength', 'spline']
    else:
        # lrt_cfg['atmosphere_file'] = os.path.join(zpt_filedir, f'atm_profiles_{date_s}_{case_tag}.dat')
        lrt_cfg['number_of_streams'] = 4
        lrt_cfg['mol_abs_param'] = 'reptran coarse'
        # ch4_file = os.path.join(zpt_filedir, f'ch4_profiles_{date_s}_{case_tag}.dat')
        input_dict_extra = {
                            'source': 'thermal',
                            'albedo_add': '0',
                            'atm_z_grid': atm_z_grid_str,
                            # 'mol_file': f'CH4 {ch4_file}',
                            # 'wavelength_grid_file': 'wvl_grid_thermal.dat',
                            'wavelength_add' : '4500 42000',
                            'output_process': 'integrate',
                            }
        Nx_effective = 1 # integrate over all wavelengths
        mute_list = ['albedo', 'wavelength', 'spline', 'source solar', 'atmosphere_file']
    #/----------------------------------------------------------------------------/#
    
    # rt setup
    #/----------------------------------------------------------------------------\#
    fdir_tmp = './tmp_dir'
    os.makedirs(fdir_tmp, exist_ok=True)
    date = dt.datetime(2024, 5, 31)
    init = er3t.rtm.lrt.lrt_init_mono_rad(
            input_file  = '%s/input.txt'  % (fdir_tmp),
            output_file = '%s/output.txt' % (fdir_tmp),
            date        = date,
            surface_albedo=0.25,
            solar_zenith_angle = 30,
            # wavelength         = wavelength,
            # Nx = Nx_effective,
            output_altitude    = 'toa',
            input_dict_extra   = input_dict_extra.copy(),
            mute_list          = mute_list,
            lrt_cfg            = lrt_cfg,
            cld_cfg            = None,
            aer_cfg            = None,
            # output_format     = 'lambda uu edir edn',
            # output_process= 'per_band',
            )
    #\----------------------------------------------------------------------------/#
    
    er3t.rtm.lrt.lrt_run(init)        
    # data = er3t.rtm.lrt.lrt_read_uvspec_rad([init])
    output = pd.read_csv(f'{fdir_tmp}/output.txt', delim_whitespace=True, comment='#', header=None)
    output.columns = ['wavelength', 'radiance', ]
    output['weight'] = weight
    print(solar_g)
    print(output)
    
    final = 0
    final_solar = 0
    for gi in range(g_number):
        final += output.iloc[gi, 1] * weight[gi]
        final_solar += solar_g[gi] * weight[gi]
    print("Final radiance:", final)
    print("Final solar flux:", final_solar * np.cos(30*np.pi/180))
    print("reflectivity:", final / (final_solar * np.cos(30*np.pi/180)) )
    

    
    with open(f'g_tau_solar_weight_sfc_{modis}_{band}_{Ng}_{alpha}_{m}.pkl', 'rb') as f:
        g_tau_solar_weight = pkl.load(f)
    levels = g_tau_solar_weight['levels']
    tau_g_total, solar_g, weight = g_tau_solar_weight['tau_g_total'], g_tau_solar_weight['solar_g'], g_tau_solar_weight['weight']
    g_number = len(weight)
    # xx_wvl_grid = np.arange(360, 1990.1, 5)
    with open(os.path.join('.', f'tau_g_total_{modis}_{band}_{Ng}.dat'), 'w') as f_grid:
        ### Build all profile lines in one go.
        # header = ('# levels ' + ' '.join(['%12d' % gi for gi in range(g_number)]) + '\n'
        #             )
        header = (' '.join(['%12.6e' % levels[i] for i in range(len(levels))[::-1]]) + '\n'
        )
        lines = [
                 f'{(gi+1.0)*100:12.2f} ' + ' '.join(['%12.6e' % tau_g_total[i, gi] for i in range(len(levels)-1)[::-1]])
                 for gi in range(g_number)
                ]
        f_grid.write(header + "\n".join(lines))
        # f_grid.write("\n".join(lines))
    
    with open('solar_g.dat', 'w') as f_solar:
            header = ('# SSFR version solar flux\n'
                    '# wavelength (nm)      flux (mW/m^2/nm)\n'
                    )
            # Build all profile lines in one go.
            lines = [
                    f'{(gi+1.0)*100:12.6f} '
                    f'{solar_g[gi]:12.6e}'
                    for gi in range(g_number)
                    ]
            f_solar.write(header + "\n".join(lines))
    

    atm_z_grid = levels
    z_list = atm_z_grid
    atm_z_grid_str = ' '.join(['%.2f' % z for z in atm_z_grid])
    # rt initialization
    #/----------------------------------------------------------------------------\#
    lrt_cfg = er3t.rtm.lrt.get_lrt_cfg()
    zpt_filedir = './zpt_dir'
    if not lw:
        lrt_cfg['atmosphere_file'] = f'{zpt_filedir}/atm_profiles_main_{modis}_{band}.dat'
        # lrt_cfg['atmosphere_file'] = lrt_cfg['atmosphere_file'].replace('afglus.dat', 'afglss.dat')
        # lrt_cfg['solar_file'] = None
        lrt_cfg['solar_file'] = None
        lrt_cfg['solar_file'] = 'solar_g.dat'
        lrt_cfg['number_of_streams'] = 16
        # lrt_cfg['mol_abs_param'] = 'reptran coarse'
        # lrt_cfg['mol_abs_param'] = f'reptran medium'
        lrt_cfg['output_process'] = 'per_band'
        input_dict_extra = {
                            # 'crs_model': 'rayleigh Bodhaine29',
                            # 'crs_model': 'rayleigh Nicolet',
                            # 'crs_model': 'o3 Bogumil',
                            # 'atm_z_grid': atm_z_grid_str,
                            'source': 'solar '+'solar_g.dat'+' per_band',
                            'mol_tau_file': 'abs ' + os.path.join('.', f'tau_g_total_{modis}_{band}_{Ng}.dat'),
                            # 'no_scattering':'mol',
                            # 'no_absorption':'mol',
                            # 'output_quantity': 'reflectivity',
                            # 'output_process': 'per_band',
                            }
        Nx_effective = g_number
        mute_list = ['solar_file', 'wavelength', 'spline']
    else:
        # lrt_cfg['atmosphere_file'] = os.path.join(zpt_filedir, f'atm_profiles_{date_s}_{case_tag}.dat')
        lrt_cfg['number_of_streams'] = 4
        lrt_cfg['mol_abs_param'] = 'reptran coarse'
        # ch4_file = os.path.join(zpt_filedir, f'ch4_profiles_{date_s}_{case_tag}.dat')
        input_dict_extra = {
                            'source': 'thermal',
                            'albedo_add': '0',
                            'atm_z_grid': atm_z_grid_str,
                            # 'mol_file': f'CH4 {ch4_file}',
                            # 'wavelength_grid_file': 'wvl_grid_thermal.dat',
                            'wavelength_add' : '4500 42000',
                            'output_process': 'integrate',
                            }
        Nx_effective = 1 # integrate over all wavelengths
        mute_list = ['albedo', 'wavelength', 'spline', 'source solar', 'atmosphere_file']
    #/----------------------------------------------------------------------------/#
    
    # rt setup
    #/----------------------------------------------------------------------------\#
    fdir_tmp = './tmp_dir'
    os.makedirs(fdir_tmp, exist_ok=True)
    date = dt.datetime(2024, 5, 31)
    init = er3t.rtm.lrt.lrt_init_mono_rad(
            input_file  = '%s/input.txt'  % (fdir_tmp),
            output_file = '%s/output.txt' % (fdir_tmp),
            date        = date,
            surface_albedo=0.25,
            solar_zenith_angle = 30,
            # wavelength         = wavelength,
            # Nx = Nx_effective,
            output_altitude    = 'toa',
            input_dict_extra   = input_dict_extra.copy(),
            mute_list          = mute_list,
            lrt_cfg            = lrt_cfg,
            cld_cfg            = None,
            aer_cfg            = None,
            # output_format     = 'lambda uu edir edn',
            # output_process= 'per_band',
            )
    #\----------------------------------------------------------------------------/#
    
    er3t.rtm.lrt.lrt_run(init)        
    # data = er3t.rtm.lrt.lrt_read_uvspec_rad([init])
    output = pd.read_csv(f'{fdir_tmp}/output.txt', delim_whitespace=True, comment='#', header=None)
    output.columns = ['wavelength', 'radiance', ]
    output['weight'] = weight
    print(solar_g)
    print(output)
    
    final_sfc_order = 0
    final_solar_sfc_order = 0
    for gi in range(g_number):
        final_sfc_order += output.iloc[gi, 1] * weight[gi]
        final_solar_sfc_order += solar_g[gi] * weight[gi]
    print("Final radiance:", final_sfc_order)
    print("Final solar flux:", final_solar_sfc_order * np.cos(30*np.pi/180))
    print("reflectivity:", final_sfc_order / (final_solar_sfc_order * np.cos(30*np.pi/180)) )
    
    if modis == 'terra' and band == 15:
        rep_channel = 'modis_terra_b06'
    elif modis == 'terra' and band == 16:
        rep_channel = 'modis_terra_b07'
    elif modis == 'terra' and band == 7:
        rep_channel = 'modis_terra_b04'
    elif modis == 'terra' and band == 12:
        rep_channel = 'modis_terra_b02'
    elif modis == 'terra' and band == 8:
        rep_channel = 'modis_terra_b01'
    elif modis == 'terra' and band == 3:
        rep_channel = 'modis_terra_b03'
    
    # rt initialization
    #/----------------------------------------------------------------------------\#
    lrt_cfg = er3t.rtm.lrt.get_lrt_cfg()
    zpt_filedir = './zpt_dir'
    if not lw:
        lrt_cfg['atmosphere_file'] = f'{zpt_filedir}/atm_profiles_main_{modis}_{band}.dat'
        # lrt_cfg['atmosphere_file'] = lrt_cfg['atmosphere_file'].replace('afglus.dat', 'afglss.dat')
        lrt_cfg['solar_file'] = None
        lrt_cfg['number_of_streams'] = 16
        # lrt_cfg['mol_abs_param'] = 'reptran coarse'
        lrt_cfg['mol_abs_param'] = f'reptran_channel {rep_channel}'
        lrt_cfg['output_process'] = 'integrate'
        input_dict_extra = {
                            # 'crs_model': 'rayleigh Bodhaine29',
                            # 'crs_model': 'rayleigh Nicolet',
                            # 'crs_model': 'o3 Bogumil',
                            'atm_z_grid': atm_z_grid_str,
                            # 'source': 'solar '+'solar_g.dat'+' per_band',
                             'source': 'solar',
                            # 'mol_tau_file': 'abs ' + os.path.join('.', f'tau_g_total_{modis}_{band}.dat'),
                            # 'no_scattering':'mol',
                            # 'no_absorption':'mol',
                            # 'output_quantity': 'reflectivity',
                            # 'output_process': 'per_band',
                            # 'wavelength_add': '841 876',
                            # 'wavelength_add': '1628 1652',
                            }
        Nx_effective = g_number
        mute_list = ['solar_file', 'spline', 'wavelength', 'slit_function_file']
        # mute_list = []
    else:
        # lrt_cfg['atmosphere_file'] = os.path.join(zpt_filedir, f'atm_profiles_{date_s}_{case_tag}.dat')
        lrt_cfg['number_of_streams'] = 4
        lrt_cfg['mol_abs_param'] = 'reptran coarse'
        # ch4_file = os.path.join(zpt_filedir, f'ch4_profiles_{date_s}_{case_tag}.dat')
        input_dict_extra = {
                            'source': 'thermal',
                            'albedo_add': '0',
                            'atm_z_grid': atm_z_grid_str,
                            # 'mol_file': f'CH4 {ch4_file}',
                            # 'wavelength_grid_file': 'wvl_grid_thermal.dat',
                            'wavelength_add' : '4500 42000',
                            'output_process': 'integrate',
                            }
        Nx_effective = 1 # integrate over all wavelengths
        mute_list = ['albedo', 'wavelength', 'spline', 'source solar', 'atmosphere_file']
    #/----------------------------------------------------------------------------/#
    
    # rt setup
    #/----------------------------------------------------------------------------\#
    fdir_tmp = './tmp_dir'
    os.makedirs(fdir_tmp, exist_ok=True)
    date = dt.datetime(2024, 5, 31)
    init = er3t.rtm.lrt.lrt_init_mono_rad(
            input_file  = '%s/input_reptran.txt'  % (fdir_tmp),
            output_file = '%s/output_reptran.txt' % (fdir_tmp),
            date        = date,
            surface_albedo=0.25,
            solar_zenith_angle = 30,
            # wavelength         = 860,
            # Nx = Nx_effective,
            output_altitude    = 'toa',
            input_dict_extra   = input_dict_extra.copy(),
            mute_list          = mute_list,
            lrt_cfg            = lrt_cfg,
            cld_cfg            = None,
            aer_cfg            = None,
            output_format     = 'lambda uu edir edn',
            # output_process= 'per_band',
            )
    #\----------------------------------------------------------------------------/#
    
    er3t.rtm.lrt.lrt_run(init)        
    data = er3t.rtm.lrt.lrt_read_uvspec_rad_toa([init])
    print((data.rad/(data.toa)).flatten())
    reptran_rad = data.rad.flatten()[0]*1000
    reptran_toa = data.toa.flatten()[0]*1000
    reptran_ref = reptran_rad/reptran_toa
    
    output_df_file = 'results_lrt_test.csv'
    if os.path.exists(output_df_file):
        with open(output_df_file, 'rb') as f:
            output_df = pd.read_csv(f)
        new_row = {
            'modis': modis, 
            'band': band, 
            'Ng': Ng, 
            'alpha': alpha,
            'm': m,
            'sckd_rad': final, 
            'sckd_toa': final_solar * np.cos(30*np.pi/180), 
            'sckd_ref': final / (final_solar * np.cos(30*np.pi/180)),
            'sckd_rad_sfc_order': final_sfc_order, 
            'sckd_toa_sfc_order': final_solar_sfc_order * np.cos(30*np.pi/180), 
            'sckd_ref_sfc_order': final_sfc_order / (final_solar_sfc_order * np.cos(30*np.pi/180)),
            'reptran_rad': reptran_rad,
            'reptran_toa': reptran_toa,
            'reptran_ref': reptran_ref,
        }
        output_df = output_df.append(new_row, ignore_index=True)
        output_df.to_csv(output_df_file, index=False)
        print("Updated existing output DataFrame with new results.")
    else:
        output_df = pd.DataFrame([
            {'modis': modis, 'band': band, 'Ng': Ng, 'alpha': alpha, 'm': m,
             'sckd_rad': final, 
             'sckd_toa': final_solar * np.cos(30*np.pi/180), 
             'sckd_ref': final / (final_solar * np.cos(30*np.pi/180)),
             'sckd_rad_sfc_order': final_sfc_order, 
             'sckd_toa_sfc_order': final_solar_sfc_order * np.cos(30*np.pi/180), 
             'sckd_ref_sfc_order': final_sfc_order / (final_solar_sfc_order * np.cos(30*np.pi/180)),
             'reptran_rad': reptran_rad,
             'reptran_toa': reptran_toa,
             'reptran_ref': reptran_ref,
             }
        ])
        output_df.to_csv(output_df_file, index=False)
        print("Created new output DataFrame with results.")
        

def plot_results(output_df_file='results_lrt_test.csv'):
    import matplotlib.pyplot as plt
    import pandas as pd
    
    output_df = pd.read_csv(output_df_file)
    
    for modis in output_df['modis'].unique():
        df_modis = output_df[output_df['modis'] == modis]
        for alpha in df_modis['alpha'].unique():
            df_modis_alpha = df_modis[df_modis['alpha'] == alpha]
            for m in df_modis_alpha['m'].unique():
                df_modis_m = df_modis_alpha[df_modis_alpha['m'] == m]
                for band in df_modis_m['band'].unique():
                    df_modis_band = df_modis_m[df_modis_m['band'] == band]
                    
                    # Plot SCKD results
                    # plt.figure(figsize=(10, 6))
                    # plt.bar(df_modis_band['Ng'], df_modis_band['sckd_ref'], label='SCKD Reflectivity', alpha=0.7, )
                    # plt.bar(df_modis_band['Ng'], df_modis_band['reptran_ref'], label='Reptran Reflectivity', alpha=0.7, color='orange')
                    # plt.xlabel('Number of G-Points (Ng)')
                    # plt.ylabel('Reflectivity')
                    # plt.title(f'{modis} Band {band} - Alpha {alpha}, m {m}')
                    # plt.legend()
                    # plt.grid()
                    # plt.savefig(f'results_{modis}_band_{band}_alpha_{alpha}_m_{m}.png')
                    # plt.show()
                    
                    if modis == 'terra' and band == 15:
                        wvl = 1640
                    elif modis == 'terra' and band == 16:
                        wvl = 2130
                    elif modis == 'terra' and band == 7:
                        wvl = 550
                    elif modis == 'terra' and band == 12:
                        wvl = 860
                    elif modis == 'terra' and band == 8:
                        wvl = 650
                    elif modis == 'terra' and band == 3:
                        wvl = 470
                    
                    Ng_list = ("4", "8", "16", "32")
                    output_result = {
                        'sckd (mean sorted)': df_modis_band['sckd_ref'],
                        'sckd (sfc sorted)': df_modis_band['sckd_ref_sfc_order'],
                        'reptran': df_modis_band['reptran_ref'],
                    }

                    x = np.arange(len(Ng_list))  # the label locations
                    width = 0.25  # the width of the bars
                    multiplier = 0

                    fig, ax = plt.subplots(layout='constrained')

                    for attribute, measurement in output_result.items():
                        offset = width * multiplier
                        rects = ax.bar(x + offset, measurement, width, label=attribute)
                        ax.bar_label(rects, padding=3, fmt='%.3f', fontsize=6)
                        multiplier += 1

                    # Add some text for labels, title and custom x-axis tick labels, etc.
                    ax.set_ylabel('Reflectance', fontsize=12)
                    ax.set_xlabel('Number of G-Points (Ng)', fontsize=12)
                    ax.set_title(f'{wvl} nm - alpha={alpha}, m={m}', fontsize=14)
                    ax.set_xticks(x + width, Ng_list)
                    # ax.legend(loc='upper left', ncols=3)
                    # ax.set_ylim(0, 250)
                    ax.legend()
                    
                    fig.tight_layout()
                    plt.savefig(f'results_{modis}_band_{band}_alpha_{alpha}_m_{m}.png')
                    # plt.show()
                    # sys.exit()
    

if __name__ == "__main__":
    
    # test_er3t_atm(modis='terra', band=8, Ng=16, alpha=2)
    # lrt_test(lw=False, modis='terra', band=8, Ng=16, alpha=2)
    
    # for band in [3, 7, 8, 12, 15, 16]:
    #     for Ng in [4, 8, 16, 32]:
    #         for alpha in [1.0, 1.5, 2.0, 4.0]:
    #             for m in [3, 4, 5, 7]:
    #                 test_er3t_atm(modis='terra', band=band, Ng=Ng, alpha=alpha, m=m)
    #                 lrt_test(lw=False, modis='terra', band=band, Ng=Ng, alpha=alpha, m=m)
                    
    
    plot_results(output_df_file='results_lrt_test.csv')