import numpy as np
import matplotlib.pyplot as plt
from utils import rj2cmb
from utils import cmb2rj

spectral_data = np.loadtxt('output/final_samples_cmb-synch-2mbb_cloud.cmb-synch-genmbb_nb7_seed10_30.0_500.0.dat')
cmb_I_Chain_Values = spectral_data.T[0]
cmb_Q_Chain_Values = spectral_data.T[1]
cmb_U_Chain_Values = spectral_data.T[2]

summary_stat_data = np.loadtxt('output/final_summary_cmb-synch-2mbb_cloud.cmb-synch-genmbb_nb7_seed10_cut0.dat')
N_experiments = len(cmb_I_Chain_Values)

#Indices of [mean_cmb_I, std_cmb_I, mean_cmb_q, std_cmb_q, mean_cmb_U, std_cmb_U] in Summary Array
summary_stat_data = np.take(summary_stat_data,[3,4,6,7,9,10])


cmb_I_inputed = 50
cmb_Q_inputed = 0.6
cmb_U_inputed = 0.6

cmb_I_bias = (summary_stat_data[0] - cmb_I_inputed )/ summary_stat_data[1]
cmb_Q_bias = (summary_stat_data[2] - cmb_Q_inputed )/ summary_stat_data[3]
cmb_U_bias = (summary_stat_data[4] - cmb_U_inputed )/ summary_stat_data[5]


plt.suptitle("Bias in Recovery of Stokes Parameters using SinglePixel  with " + str(N_experiments) + " Experiments" +
            "\n" + "")

plt.subplot(221)
plt.title("I$_{CMB}$ Experiment")
plt.axvline(color = "red", x = cmb_I_inputed)
plt.hist(cmb_I_Chain_Values, bins = 30)
plt.xlabel("Predicted I$_{CMB}$ in $\mu$K$_{RJ}$")
plt.ylabel("N")



plt.subplot(222)
plt.title("Q$_{CMB}$ Experiment")
plt.axvline(color = "red", x = cmb_Q_inputed)
plt.hist(cmb_Q_Chain_Values, bins = 30)
plt.xlabel("Predicted Q$_{CMB}$ in $\mu$K$_{RJ}$")
plt.ylabel("N")

plt.subplot(223)
plt.title("U$_{CMB}$ Experiment")
plt.axvline(color = "red", x = cmb_U_inputed)
plt.hist(cmb_U_Chain_Values, bins = 30)
plt.xlabel("Predicted U$_{CMB}$ in $\mu$K$_{RJ}$")
plt.ylabel("N")

plt.subplot(224)
plt.axhline(color = "black", y= 0)
plt.axis([-1,3,-1,1])
objects = ('I$_{CMB}$', 'Q$_{CMB}$', 'U$_{CMB}$')
y_pos = np.arange(len(objects))
bias_array = [cmb_I_bias, cmb_Q_bias, cmb_U_bias]
plt.bar(y_pos, bias_array, align='center', alpha=1.0)
plt.xticks(y_pos, objects)
plt.ylabel('Bias Units?')
plt.title("Bias in Recovered Polarization Parameters")

plt.tight_layout()
plt.show()
