# coding: utf-8

#This program allows us to visualize how the mean biases in the predicted CMB parameters depend on the value of fQ we use
# in our two-component dust cloud [TCCM] model.
#It works in three steps:
#1. Change parameters in model_list_experimental.py.
#2. Execute run_joint_mcmc_experimental.py (mcmc_iterations) times to execute multiple MCMC fit.
#3. Obtain average biases for various parameters values.

import numpy as np
import run_joint_mcmc_experimental as mc
import bias_calc_2 as bias
import model_list_experimental as ml
from utils import *
import matplotlib.pyplot as plt
import plot_cloud_model as cloudplot

#TCCM Parameter List: amp_I, amp_Q, amp_U, beta, dbeta, Td1, Td2, fI, fQ, fU
#Default TCCM Parameters = (rj2cmb(353e9, DUST_I/(1+fI)), rj2cmb(353e9, DUST_P/(1+fQ)), rj2cmb(353e9, DUST_P/(1+fU)), 1.6, 0.0, 20, 15.0, 1.0, 1.1, 0.8)

#MBB Parameter list : amp_I, amp_Q, amp_U, beta, T
#Default MBB Parameters = (rj2cmb(353e9, DUST_I), rj2cmb(353e9, DUST_P), rj2cmb(353e9, DUST_P), 1.6, 20.)

#Constants
DUST_I = 50.
DUST_P = 10. / 1.41
mcmc_iterations = 100
counter = 1

#Standard Arrays
seed_list = np.arange(mcmc_iterations)
index = np.argwhere(np.arange(-50, 51, 1) == -10)
fQ_array = np.delete(np.arange(-50, 51, 1), index)/10.0 #Removes fQ = -1 from the list, which would otherwise raise a divide by zero error.
#fQ_array = np.array([-5.0, -4.0, -3.0])


#Appends new parameters to modify_parameters.txt
def set_fQ_params_TCCM(fQ, fI = 1.0 , fU = 0.8):
    with open('modify_parameters.txt', 'a') as param_file:
        new_params = (rj2cmb(353e9, DUST_I/(1+fI)), rj2cmb(353e9, DUST_P/(1+fQ)), rj2cmb(353e9, DUST_P/(1+fU)), 1.6, 0.0, 20, 15.0, 1.0, fQ, 0.8)
        param_file.write(str(new_params) + "\n")
        #To input MBB model: new_params = (rj2cmb(353e9, DUST_I), rj2cmb(353e9, DUST_P), rj2cmb(353e9, DUST_P), 1.6, T)

def mean_bias(T):
    global counter

    run_I_bias = []
    run_Q_bias = []
    run_U_bias = []

    for i in range(mcmc_iterations):
        print str(counter) + "/" + str(mcmc_iterations*len(fQ_array))
        run_seed = seed_list[i]
        mc.main(run_seed)
        temp = bias.bias_array(run_seed) #OPTIONAL CUT PARAMETER FOR WHEN STEPS < 200

        run_I_bias.append(temp[0])
        run_Q_bias.append(temp[1])
        run_U_bias.append(temp[2])

        print "Current Mean I Bias " + str(np.mean(run_I_bias))
        print "Current Mean Q Bias " + str(np.mean(run_Q_bias))
        print "Current Mean U Bias " + str(np.mean(run_U_bias))
        counter = counter + 1

    return np.array([np.mean(run_I_bias) , np.mean(run_Q_bias), np.mean(run_U_bias)])

main_I_biases = []
main_Q_biases = []
main_U_biases = []

for T in fQ_array:                             #For specified fQ values:
    set_fQ_params_TCCM(T)                      #Update model parameter values in model_list_experimental via modify_parameters.txt
    mean_run_biases = mean_bias(T)             #Executes fitting and returns mean biases for each fQ value.
    main_I_biases.append(mean_run_biases[0])
    main_Q_biases.append(mean_run_biases[1])
    main_U_biases.append(mean_run_biases[2])



plt.figure(figsize=(16,9))
plt.suptitle('Mean Bias in CMB Polarization Parameters vs. $f_{Q}$'
            + "\n" + "Remaining TCCM Parameters Default"
            + "\n" + "In: TCCM Fit: MBB : %s MCMC Runs" % mcmc_iterations)


#Plot bias arrays against fQ in left column of figure.
plt.subplot(331)
plt.title('$I_{CMB}$')
plt.scatter(fQ_array, main_I_biases)
plt.xlabel("fQ")
plt.ylabel("Parameter Bias" + "\n" + "Mean: " +  str(round(np.mean(main_I_biases), 4)))
plt.grid("on")


plt.subplot(334)
plt.title('$Q_{CMB}$')
plt.scatter(fQ_array, main_Q_biases)
plt.xlabel("fQ")
plt.ylabel("Parameter Bias" + "\n" + "Mean: " +  str(round(np.mean(main_Q_biases), 4)))
plt.grid("on")

plt.subplot(337)
plt.title('$U_{CMB}$')
plt.scatter(fQ_array, main_U_biases)
plt.xlabel("fQ")
plt.ylabel("Parameter Bias" + "\n" + "Mean: " +  str(round(np.mean(main_U_biases), 4)))
plt.grid("on")

#Uses the fQ values implemented to plot TCCM SEDs.
cloudplot.wrapper(fQ_array)

#Adjusts spacing and saves figure
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
plt.savefig("figures/Two_Cloud_to_MBBModel_" + str(len(fQ_array)) + "_fQ_Values_" + str(mcmc_iterations) + "_MCMC_Iterations")
plt.show()
