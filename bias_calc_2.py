import numpy as np
#import matplotlib.pyplot as plt
from utils import rj2cmb
from utils import cmb2rj

def bias_array(current_seed, cut = 5000):
    # spectral_data = np.loadtxt('output/final_samples_cmb-synch-2mbb_cloud.cmb-synch-genmbb_nb7_seed%d_30.0_500.0.dat' % current_seed)
    # cmb_I_Chain_Values = spectral_data.T[0]
    # cmb_Q_Chain_Values = spectral_data.T[1]
    # cmb_U_Chain_Values = spectral_data.T[2]
    file_name = 'output/fqvals_summary_cmb-synch-2mbb_cloud.cmb-synch-mbb_nb7_seed%04d_cut' %current_seed + str(cut) + '.dat'
    #print file_name
    summary_stat_data = np.loadtxt(file_name)

    #N_experiments = len(cmb_I_Chain_Values)

    #Indices of [mean_cmb_I, std_cmb_I, mean_cmb_q, std_cmb_q, mean_cmb_U, std_cmb_U] in Summary Array

    summary_stats = np.take(summary_stat_data,[3,4,6,7,9,10])

    cmb_I_inputed = 50
    cmb_Q_inputed = 0.6
    cmb_U_inputed = 0.6

    cmb_I_bias = (summary_stats[0] - cmb_I_inputed )/ summary_stats[1]
    cmb_Q_bias = (summary_stats[2] - cmb_Q_inputed )/ summary_stats[3]
    cmb_U_bias = (summary_stats[4] - cmb_U_inputed )/ summary_stats[5]

    param_bias = [cmb_I_bias, cmb_Q_bias, cmb_U_bias]


    return param_bias
