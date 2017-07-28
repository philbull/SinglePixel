# coding: utf-8

# This program allows us to visualize the distribution of biases in the predicted CMB parameters for an arbitrary
# number of MCMC Runs using a single set of parameters. (In this file, we specify Td1 = 20K (its default value).)

# It works in two steps:
#1. Execute run_joint_mcmc_experimental.py (mcmc_iterations) times to execute multiple MCMC fit.
#2. Plot histrogram of mean biases generated from each MCMC run.

import numpy as np
import run_joint_mcmc_experimental as mc
import bias_calc_2 as bias
import model_list_experimental as ml
from utils import *
import matplotlib.pyplot as plt

#TCCM Parameter list : amp_I, amp_Q, amp_U, beta, dbeta, Td1, Td2, fI, fQ, fU
#Default_TCCM_Parameters = (rj2cmb(353e9, DUST_I/2.), rj2cmb(353e9, DUST_P/2.1), rj2cmb(353e9, DUST_P/1.8), 1.6, 0.0, 20.0, 15.0, 1.0, 1.1, 0.8)
#MBB Parameter list : amp_I, amp_Q, amp_U, beta, T
#Default_MBB_Parameters = (rj2cmb(353e9, DUST_I), rj2cmb(353e9, DUST_P), rj2cmb(353e9, DUST_P), 1.6, 20.)

#Constants
DUST_I = 50.
DUST_P = 10. / 1.41
mcmc_iterations = 10

#Standard Arrays
seed_list = np.arange(mcmc_iterations)

#Modify Parameters
def set_Td1_params_TCCM(T):
    with open('modify_parameters.txt', 'a') as param_file:
        TCCM_Params = (rj2cmb(353e9, DUST_I/2.), rj2cmb(353e9, DUST_P/2.1), rj2cmb(353e9, DUST_P/1.8), 1.6, 0.0, T, 15.0, 1.0, 1.1, 0.8)        #Gets MBB params to input into models list, then writes it in there
        param_file.write(str(TCCM_Params) + "\n")

set_Td1_params_TCCM(20)

def calc_bias():
    counter = 1
    run_I_bias = []
    run_Q_bias = []
    run_U_bias = []

    for i in range(mcmc_iterations):
        print str(counter) + "/" + str(mcmc_iterations)
        run_seed = seed_list[i]
        mc.main(run_seed)
        temp = bias.bias_array(run_seed)
        print "From composite analysis: ", temp

        run_I_bias.append(temp[0])
        run_Q_bias.append(temp[1])
        run_U_bias.append(temp[2])

        counter += 1

    plt.figure(figsize=(12,6))
    plt.suptitle('Mean Bias in CMB Polarization Parameters: $T_{d1}$ = 20 K'
                + "\n" + "Remaining TCCM Parameters Default"
                + "\n" + "In: TCCM Fit: MBB : %s Runs" % mcmc_iterations)

    plt.subplot(221)
    plt.title('$I_{CMB}$')
    plt.hist(run_I_bias)
    plt.axis([-8,8,0,30])
    plt.xlabel("Bias in I")
    plt.ylabel("N mean = " + str(round(np.mean(run_I_bias),4)))


    plt.subplot(222)
    plt.title('$Q_{CMB}$')
    plt.hist(run_Q_bias)
    plt.axis([-8,8,0,30])
    plt.xlabel("Bias in Q")
    plt.ylabel("N mean = " + str(round(np.mean(run_Q_bias),4)))

    plt.subplot(223)
    plt.title('$U_{CMB}$')
    plt.hist(run_U_bias)
    plt.axis([-8,8,0,30])
    plt.xlabel("Bias in U")
    plt.ylabel("N mean = " + str(round(np.mean(run_U_bias),4)))

    plt.tight_layout()
    plt.subplots_adjust(top = 0.75, hspace = 0.5)
    plt.savefig("figures/Histogram_Two_Cloud_to_MBBModel_Default_Parameters_" + str(mcmc_iterations) + "_MCMC_Iterations")
    plt.show()

calc_bias()
