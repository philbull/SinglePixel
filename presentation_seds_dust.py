#FIXME: NEEds work
import model_list_experimental
from utils import *
import numpy as np
import matplotlib.pyplot as plt
import models

ultra_counter = 1

def frame(fQ_value):
    global ultra_counter
    DUST_I = 50.
    DUST_P = 10. / 1.41

    wavelengths = np.logspace(np.log10(10e9), np.log10(1000e9), 400)
    plot_axis = wavelengths / 1e9

    #For TCCM: dust_model = models.DustGen(rj2cmb(353e9, DUST_I/2.), rj2cmb(353e9, DUST_P/2.1), rj2cmb(353e9, DUST_P/1.8) ,1.6,0.0,20.0,15.0,1.0,1.1,0.8)
    #For MBB:
    #dust_model = model_list_experimental.simple_dust_model
    TCCM_dust_model = models.DustGen(322.63746768871823, -22.882089907001294, 50.849088682225094, 1.6, 0.0, 20, 15.0, 1.0, fQ_value, 0.8)


    I_cmb_scaling_factor = 50 / cmb2rj(30e9, model_list_experimental.cmb_model.scaling([30e9])[0]) #Applies figure frequencies from Table 1
    I_sync_scaling_factor = 30 / cmb2rj(30e9, model_list_experimental.sync_model.scaling(30e9)[0]) #Applies figure frequencies from Table 1
    I_ff_scaling_factor = 30 / cmb2rj(30e9, model_list_experimental.ff_model.scaling(30e9)[0]) #Applies figure frequencies from Table 1
    I_ame_scaling_factor = 30 / cmb2rj(30e9, model_list_experimental.ame_model.scaling(30e9)[0]) #Applies figure frequencies from Table 1
    I_dust_scaling_factor = 50 / cmb2rj(353e9, model_list_experimental.simple_dust_model.scaling(353e9)[0]) #Applies figure frequencies from Table 1
    I_TCCM_scaling_factor = 50 / cmb2rj(353e9, TCCM_dust_model.scaling(353e9)[0])

    Q_cmb_scaling_factor = 0.6 / cmb2rj(30e9, model_list_experimental.cmb_model.scaling([30e9])[1]) #Applies figure frequencies from Table 1
    Q_sync_scaling_factor = 10 / cmb2rj(30e9, model_list_experimental.sync_model.scaling(30e9)[1]) #Applies figure frequencies from Table 1
    Q_dust_scaling_factor = 3.5 / cmb2rj(353e9, model_list_experimental.simple_dust_model.scaling(353e9)[1]) #Applies figure frequencies from Table 1
    Q_TCCM_scaling_factor = 3.5 / cmb2rj(353e9, TCCM_dust_model.scaling(353e9)[1])

    plt.figure(figsize = (12,5))
    plt.suptitle("Foreground Component SEDs: fQ = " + str(fQ_value), size = 30, y = 0.95)

    #Plot Intensity

    plt.subplot(121)
    # plt.plot(plot_axis, I_cmb_scaling_factor * cmb2rj(wavelengths, model_list_experimental.cmb_model.scaling(wavelengths)[0]), label = "CMB")
    # plt.plot(plot_axis, I_sync_scaling_factor * cmb2rj(wavelengths, model_list_experimental.sync_model.scaling(wavelengths)[0]), label = "Synchrotron")
    # plt.plot(plot_axis, I_ff_scaling_factor * cmb2rj(wavelengths, model_list_experimental.ff_model.scaling(wavelengths)[0]), label = "Free-free")
    # plt.plot(plot_axis, I_ame_scaling_factor * cmb2rj(wavelengths, model_list_experimental.ame_model.scaling(wavelengths)[0]), label = "AME")
    plt.plot(plot_axis, I_dust_scaling_factor * cmb2rj(wavelengths, model_list_experimental.simple_dust_model.scaling(wavelengths)[0]), label = "MBB Dust Model",  color = "red")
    plt.plot(plot_axis, I_TCCM_scaling_factor * cmb2rj(wavelengths, TCCM_dust_model.scaling(wavelengths)[0]), label = "2MBB Dust Model",  color = "blue")

    # plt.plot(plot_axis,    (
    #                         I_cmb_scaling_factor * cmb2rj(wavelengths, model_list_experimental.cmb_model.scaling(wavelengths)[0]) +
    #                         I_sync_scaling_factor * cmb2rj(wavelengths, model_list_experimental.sync_model.scaling(wavelengths)[0]) +
    #                         I_ff_scaling_factor * cmb2rj(wavelengths, model_list_experimental.ff_model.scaling(wavelengths)[0]) +
    #                         I_ame_scaling_factor * cmb2rj(wavelengths, model_list_experimental.ame_model.scaling(wavelengths)[0]) +
    #                         I_dust_scaling_factor * cmb2rj(wavelengths, model_list_experimental.simple_dust_model.scaling(wavelengths)[0])
    #                         ), ls = "dashed", label = "Sum", color = "brown")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("$\\nu$ (GHz)", size = 20)
    plt.ylabel('I ($\\mu$$K_{RJ}$)', size = 20)
    plt.xlim(1e1,1e3)
    plt.ylim(1e-1, 1e3)
    plt.legend(loc = "lower right")
    plt.tick_params(axis='both', which='major', labelsize=15)


     #Plot Q Polarization

    plt.subplot(122)
    # plt.plot(plot_axis, Q_cmb_scaling_factor * cmb2rj(wavelengths, model_list_experimental.cmb_model.scaling(wavelengths)[1]), label = "CMB")
    # plt.plot(plot_axis, Q_sync_scaling_factor * cmb2rj(wavelengths, model_list_experimental.sync_model.scaling(wavelengths)[1]), label = "Synchrotron")
    plt.plot(plot_axis, Q_dust_scaling_factor * cmb2rj(wavelengths, model_list_experimental.simple_dust_model.scaling(wavelengths)[1]), label = "MBB Dust Model", color = "red")
    plt.plot(plot_axis, Q_TCCM_scaling_factor * cmb2rj(wavelengths, TCCM_dust_model.scaling(wavelengths)[1]), label = "2MBB Dust Model", color = "blue")

    # plt.plot(plot_axis, (
    #                     Q_cmb_scaling_factor * cmb2rj(wavelengths, model_list_experimental.cmb_model.scaling(wavelengths)[1]) +
    #                     Q_sync_scaling_factor * cmb2rj(wavelengths, model_list_experimental.sync_model.scaling(wavelengths)[1]) +
    #                     Q_dust_scaling_factor * cmb2rj(wavelengths, model_list_experimental.simple_dust_model.scaling(wavelengths)[1])
    #                     ), ls = "dashed", label = "Sum", color = "brown")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("$\\nu$ (GHz)", size = 20)
    plt.ylabel('Q ($\\mu$$K_{RJ}$)', size = 20)
    plt.xlim(1e1,1e3)
    plt.ylim(1e-1, 1e3)

    plt.tick_params(axis='both', which='major', labelsize=15)

    plt.tight_layout()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=0.8, wspace=0.3, hspace=0.5)
    plt.legend(loc = "lower right")
    plt.savefig("animation/" + str(ultra_counter)+".png")
    ultra_counter += 1

index = np.argwhere(np.arange(-50, 51, 1) == -10)
fQ_array = np.delete(np.arange(-50, 51, 1), index)/10.0

for i in fQ_array:
    frame(i)
