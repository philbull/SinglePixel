import matplotlib.pyplot as plt
import numpy as np
import models
from utils import rj2cmb
import random

'''
add conversions for
rj
physical (Jansky / strd)
cmbtemp 
'''

DUST_I = 50.
DUST_P = 10. / 1.41

PRECISION = 400
curves_summed = 5
sums = 2

nu_Hz = np.logspace(np.log10(1e9), np.log10(800e9), PRECISION)
nu_GHz = nu_Hz / 1.0e9


#CMB model from model_list.py
cmb_model = models.CMB( amp_I=50., amp_Q=0.6, amp_U=0.6 ) #Originally 50,0.6,0.6

#Synchrotron model from model_list.py
#sync_model = models.SyncPow( amp_I=30., amp_Q=10., amp_U=10., sync_beta=-3.2 ) #Originally 30,10,10,-3.2

#Modified Black Body f(B,T) dust model
def update_dust_var(variable_B, variable_T):
    dust_model = models.DustMBB( amp_I=rj2cmb(353e9, DUST_I), #Imported from model_list.py
                                 amp_Q=rj2cmb(353e9, DUST_P),
                                 amp_U=rj2cmb(353e9, DUST_P),
                                 dust_beta= variable_B, dust_T = variable_T) #Originally 1.6, 20K
    return dust_model.scaling(nu_Hz)[0]


#records median,max,min B,T tested
universal_set_betas = []
universal_set_T = []

for i in range(sums):
    cumulative_dust_I = np.zeros(PRECISION)
    single_composite_T = []
    single_composite_betas = []

    for j in range(curves_summed):
        variable_B = np.random.uniform(1.0,2.4)
        variable_T = np.random.uniform(10.0,30.0)

        single_composite_T.append(variable_T)
        single_composite_betas.append(variable_B)

        cumulative_dust_I += update_dust_var(variable_B, variable_T)

    universal_set_T.append(single_composite_T)
    universal_set_betas.append(single_composite_betas)
    plt.plot(nu_GHz, cumulative_dust_I/curves_summed)

#Plot curves from B,T statistics from universal set
median_B = np.median(universal_set_betas)
median_T = np.median(universal_set_T)
min_T = np.min(universal_set_T)
max_T= np.max(universal_set_T)
min_B = np.min(universal_set_betas)
max_B= np.max(universal_set_betas)

plt.plot(nu_GHz, update_dust_var(max_B, max_T), "r--", linewidth = 2) #max B with max as red dashed
plt.plot(nu_GHz, update_dust_var(min_B, min_T), "g--", linewidth = 2) #min B with min T green dashes
#medianline = plt.plot(nu_GHz, update_dust_var(median_B, median_T), "black", linewidth = 4 , label = "medianline") #median

#plt.legend(handles=[medianline])


#Plots CMB
plt.plot(nu_GHz, cmb_model.scaling(nu_Hz)[0])


#Plot Setup
plt.grid("on")
plt.title("MBB Dust Model\n" + str(sums) + " Arithmetic Means of " +  str(curves_summed) + " MBB Dust Spectra\n" + str(PRECISION) + " Frequency Samples")
plt.xlabel(" $\\nu$ (GHz)")
plt.ylabel("Intensity of Light (Units?)")
plt.yscale('log')
plt.xscale('log')
plt.show()
