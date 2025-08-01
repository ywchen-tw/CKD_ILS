import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import er3t
from importmonkey import add_path
add_path("/Users/yuch8913/programming/CKD_ILS/")
import sckd

def main():
    # This function is just a placeholder to allow the script to run as a module


    fname_atm = '/Users/yuch8913/programming/er3t/er3t_mca_v11/er3t/examples/tmp-data/00_er3t_mca/example_05_rad_les_cloud_3d/atm.pk'

    levels = np.arange(0.0, 40.1, 4)

    atm0   = er3t.pre.atm.atm_atmmod(levels=levels, fname=fname_atm, fname_atmmod=er3t.common.params['atmospheric_profile'], overwrite=False)

    wv1abs = 762.0
    wv2abs = 778.0
    slit_wvl = np.arange(wv1abs, wv2abs+1e-7, 0.1)
    slit_response = np.ones_like(slit_wvl)

    ssfr_slit_vis_file = "/Users/yuch8913/programming/er3t/er3t/er3t/data/slit/ssfr/vis_0.1nm_s.dat"
    slit_data = sckd.read_dat(ssfr_slit_vis_file)
    slit_response = np.interp(slit_wvl-770, slit_data[:, 0], slit_data[:, 1])

    if 1:#not os.path.exists('tmp_abs.pkl'):
        nu_final, cont_tau_final, lbl_tau_final, solar_data_interpolate, slit_response_final, ind_sort = sckd.abs_tau_calc(atm0=atm0,
                                wv1abs=wv1abs, wv2abs=wv2abs, dvabs=0.01, 
                                slit_wvl=slit_wvl, slit_response=slit_response,
                                radflag=True)
        
        # save nu, cont_tau_final, lbl_tau_final, solar_data_interpolate, slit_response_final, ind_sort
        with open('tmp_abs.pkl', 'wb') as f:
            pkl.dump((nu_final, cont_tau_final, lbl_tau_final, solar_data_interpolate, slit_response_final, ind_sort), f)
    else:
        with open('tmp_abs.pkl', 'rb') as f:
            nu_final, cont_tau_final, lbl_tau_final, solar_data_interpolate, slit_response_final, ind_sort = pkl.load(f)


    # print("cont_tau_final iz=0:", cont_tau_final[0, :])
    # print("lbl_tau_final iz=0:", lbl_tau_final[0, :])
    # print(1e7 / nu_final)
    # sys.exit()
    final_tau = (cont_tau_final + lbl_tau_final) * slit_response_final
    tau_g_total, solar_g, weight = sckd.g_distribution(1e7/nu_final, final_tau, solar_data_interpolate, ind_sort, g_num=16)

    # for iz in range(tau_g_total.shape[0]):
    #     print(f"Layer {iz}: ")
    #     print("tau_g:", tau_g_total[iz, :])
    # sys.exit()

    wvl_nm = 1e7 / nu_final  # Convert wavenumber to wavelength in nm
    print('max abs:', cont_tau_final.max(), cont_tau_final.max())
    for iz in range(cont_tau_final.shape[0]):
        plt.plot(wvl_nm, cont_tau_final[iz, :], label='z={:01d}'.format(iz))
    # plt.plot(wvl_nm, ds[1], label='Foreign Absorption')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Absorption optical depth')
    plt.legend()
    plt.yscale('log')
    plt.show()

    for iz in range(lbl_tau_final.shape[0]):
        plt.plot(wvl_nm, lbl_tau_final[iz, :], label='z={:01d}'.format(iz))
    # plt.plot(wvl_nm, ds[1], label='Foreign Absorption')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('LBL Absorption optical depth')
    plt.legend()
    plt.yscale('log')
    plt.show()

    for iz in range(1):
        plt.plot(wvl_nm, final_tau[iz, :], label='z={:01d}'.format(iz))
    # plt.plot(wvl_nm, ds[1], label='Foreign Absorption')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Absorption optical depth at 1st layer')
    plt.legend()
    plt.yscale('log')
    plt.show()

    for iz in range(lbl_tau_final.shape[0]):
        plt.plot(wvl_nm, final_tau[iz, :], label='z={:01d}'.format(iz))
    # plt.plot(wvl_nm, ds[1], label='Foreign Absorption')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Absorption optical depth at each layer')
    plt.legend()
    plt.yscale('log')
    plt.show()


    plt.plot(wvl_nm, np.sum(lbl_tau_final[:, :]+cont_tau_final[:, :], axis=0))
    # plt.plot(wvl_nm, ds[1], label='Foreign Absorption')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Sum Absorption optical depth for all layers')
    plt.legend()
    plt.yscale('log')
    plt.show()


if __name__ == "__main__":
    main()