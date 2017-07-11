import matplotlib.pyplot as plt
import numpy as np
import models
from utils import rj2cmb
from utils import cmb2rj
import random

'''
add conversions for
rj
physical (Jansky / strd)
cmbtemp

#FIX DUST LINE
'''

FF_I, SYNC_I , AME_I = 30, 30., 30.
CMB_I = 50
DUST_I = 50

DUST_P = 10. / 1.41

CMB_Q = 0.6
SYNC_Q = 10
DUST_Q = 3.5

PRECISION = 400
nu_Hz = np.logspace(np.log10(1e10), np.log10(1000e9), PRECISION)
nu_GHz = nu_Hz / 1.0e9


def update_dust_var(variable_B, variable_T):
    dust_model = models.DustMBB( amp_I=rj2cmb(353e9, DUST_I),
                                 amp_Q=rj2cmb(353e9, DUST_P),
                                 amp_U=rj2cmb(353e9, DUST_P),
                                 dust_beta=variable_B, dust_T=variable_T ) #B,T originally 1.6, 20
    return np.array(dust_model.scaling(nu_Hz)) # * DUST_I


def update_ame_var(variable_nu_peak):
    ame_model = models.AMEModel( amp_I=rj2cmb(30e9, 30.),
                                 amp_Q=rj2cmb(30e9, 0.),
                                 amp_U=rj2cmb(30e9, 0.),
                                 nu_peak=variable_nu_peak ) #nu_peak originally 25
    return np.array(ame_model.scaling(nu_Hz)) * AME_I

def update_ff_var(variable_B):
    ff_model = models.FreeFreeUnpol( amp_I=rj2cmb(30e9, 30.),
                                     amp_Q=rj2cmb(30e9, 0.),
                                     amp_U=rj2cmb(30e9, 0.),
                                     ff_beta=variable_B )   #ff_beta originally -0.118
    return np.array(ff_model.scaling(nu_Hz)) * FF_I

def update_sync_var(variable_B):
    sync_model = models.SyncPow( amp_I=30., amp_Q=10., amp_U=10., sync_beta=variable_B ) #sync_beta_originally -3.2
    return np.array(sync_model.scaling(nu_Hz)) * SYNC_I

def return_cmb_spectrum():
    cmb_model = models.CMB( amp_I=50., amp_Q=0.6, amp_U=0.6 )
    return np.array(cmb_model.scaling(nu_Hz)) * CMB_I

DUST_I_RJ = rj2cmb(353e9, DUST_I)


list_index = 0

default_params_cmb = return_cmb_spectrum()[list_index]
default_params_sync_var = update_sync_var(-3.2)[list_index]
default_params_ff = update_ff_var(-0.118)[list_index]
default_params_AME = update_ame_var(25)[list_index]
default_params_dust_var = update_dust_var(1.6, 20)[list_index] * DUST_I_RJ

if list_index == 0:
    y_axis_label = "Intensity ($\mu$K$_{RJ}$)"
elif list_index == 1:
     y_axis_label = "Intensity of Q Polarization ($\mu$K$_{RJ}$)"
     default_params_ff = np.zeros(PRECISION)
     default_params_AME = np.zeros(PRECISION)


elif list_index == 2:
    y_axis_label = "Intensity of U Polarization($\mu$K$_{RJ}$)"
    default_params_ff = np.zeros(PRECISION)
    default_params_AME = np.zeros(PRECISION)


sum_spectra = (cmb2rj(nu_Hz, default_params_cmb) + cmb2rj(nu_Hz, default_params_sync_var) +
                cmb2rj(nu_Hz, default_params_ff) + cmb2rj(nu_Hz, default_params_AME) +
                cmb2rj(nu_Hz, default_params_dust_var))



plt.plot(nu_GHz, cmb2rj(nu_Hz, default_params_cmb), "black", linewidth = 3, label = "CMB")
plt.plot(nu_GHz, cmb2rj(nu_Hz, default_params_sync_var), "green" , linewidth = 3,label = "Synchrotron")
plt.plot(nu_GHz, cmb2rj(nu_Hz, default_params_ff), "blue", linewidth = 3,label = "Free-Free")
plt.plot(nu_GHz, cmb2rj(nu_Hz, default_params_AME), "yellow", linewidth = 3,label = "AME")
plt.plot(nu_GHz, cmb2rj(nu_Hz, default_params_dust_var), "red", linewidth = 3, label = "MBB Dust")
#plt.plot(nu_GHz, sum_spectra, label = "Sum of other Curves")



#Plot Setup
plt.grid("on")
plt.title("Foreground Component Simulation" + "\n" + str(PRECISION) + " Frequency Samples")
plt.xlabel(" $\\nu$ (GHz)")
plt.ylabel(y_axis_label)

plt.axis([np.min(nu_GHz), np.max(nu_GHz), np.min(update_dust_var(1.6, 20)[0]), np.max(sum_spectra)])
plt.yscale('log')
plt.xscale('log')
plt.legend(loc = "lower left")
plt.show()

















































# #CMB model from model_list.py
# cmb_model = models.CMB( amp_I=50., amp_Q=0.6, amp_U=0.6 ) #Originally 50,0.6,0.6
#
# #Synchrotron model from model_list.py
# sync_model = models.SyncPow( amp_I=30., amp_Q=10., amp_U=10., sync_beta=-3.2 ) #Originally 30,10,10,-3.2
#
# #Modified Black Body f(B,T) dust model
# def update_dust_var(variable_B, variable_T):
#     dust_model = models.DustMBB( amp_I=rj2cmb(353e9, DUST_I), #Imported from model_list.py
#                                  amp_Q=rj2cmb(353e9, DUST_P),
#                                  amp_U=rj2cmb(353e9, DUST_P),
#                                  dust_beta= variable_B, dust_T = variable_T) #Originally 1.6, 20K
#     return dust_model.scaling(nu_Hz)
#
#
# #records median,max,min B,T tested
# universal_set_betas = []
# universal_set_T = []
#
# for i in range(sums):
#     cumulative_dust_I = np.zeros(PRECISION)
#     single_composite_T = []
#     single_composite_betas = []
#
#     for j in range(curves_summed):
#         variable_B = np.random.uniform(1.0,2.4)
#         variable_T = np.random.uniform(10.0,30.0)
#
#
#
#         single_composite_T.append(variable_T)
#         single_composite_betas.append(variable_B)
#
#         cumulative_dust_I += update_dust_var(variable_B, variable_T)[0]
#
#     universal_set_T.append(single_composite_T)
#     universal_set_betas.append(single_composite_betas)
#
#     plt.plot(nu_GHz, cumulative_dust_I/curves_summed)
#
#
#
#
# #Plot curves from B,T statistics from universal set
# median_B = np.median(universal_set_betas)
# median_T = np.median(universal_set_T)
# min_T = np.min(universal_set_T)
# max_T= np.max(universal_set_T)
# min_B = np.min(universal_set_betas)
# max_B= np.max(universal_set_betas)
#
# for i in range(len(update_dust_var(max_B, max_T))):
#     plt.plot(nu_GHz, update_dust_var(max_B, max_T)[i], "r--", linewidth = 2, label = "Max B,T") #max B with max as red dashed
#     plt.plot(nu_GHz, update_dust_var(min_B, min_T)[i], "g--", linewidth = 2, label = "Min B,T") #min B with min T green dashes
#     plt.plot(nu_GHz, update_dust_var(median_B, median_T)[i], "black", linewidth = 4 , label = "Median B, T") #median
#
#
#
#
#
# #Plots CMB
# plt.plot(nu_GHz, cmb_model.scaling(nu_Hz)[0])
#
#
# #Plot Setup
# plt.grid("on")
# plt.title("MBB Dust Model\n" + str(sums) + " Arithmetic Means of " +  str(curves_summed) + " MBB Dust Spectra\n" + str(PRECISION) + " Frequency Samples")
# plt.xlabel(" $\\nu$ (GHz)")
# plt.ylabel("T $_{CMB}$")
# plt.yscale('log')
# plt.xscale('log')
# plt.legend(loc = "lower right")
# plt.show()
