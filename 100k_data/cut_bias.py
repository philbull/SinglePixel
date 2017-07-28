import numpy as np

def bias_array(cut):
    summary_stat_data = np.loadtxt('100k_data/final_summary_cmb-synch-2mbb_cloud.cmb-synch-mbb_nb7_seed0_cut%s.dat' %str(cut))

    #Generates [mean_cmb_I, std_cmb_I, mean_cmb_Q, std_cmb_Q, mean_cmb_U, std_cmb_U] from summary stat file.
    summary_stats = np.take(summary_stat_data,[3,4,6,7,9,10])

    cmb_I_entered = 50
    cmb_Q_entered = 0.6
    cmb_U_entered = 0.6

    cmb_I_bias = (summary_stats[0] - cmb_I_entered )/ summary_stats[1]
    cmb_Q_bias = (summary_stats[2] - cmb_Q_entered )/ summary_stats[3]
    cmb_U_bias = (summary_stats[4] - cmb_U_entered )/ summary_stats[5]

    param_bias = [cmb_I_bias, cmb_Q_bias, cmb_U_bias]
    return param_bias
