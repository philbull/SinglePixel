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
from mpi4py import MPI

# Set-up MPI
comm = MPI.COMM_WORLD
myid = comm.Get_rank()
nproc = comm.Get_size()

#TCCM Parameter List: amp_I, amp_Q, amp_U, beta, dbeta, Td1, Td2, fI, fQ, fU
#Default TCCM Parameters = (rj2cmb(353e9, DUST_I/(1+fI)), rj2cmb(353e9, DUST_P/(1+fQ)), rj2cmb(353e9, DUST_P/(1+fU)), 1.6, 0.0, 20, 15.0, 1.0, 1.1, 0.8)

#MBB Parameter list : amp_I, amp_Q, amp_U, beta, T
#Default MBB Parameters = (rj2cmb(353e9, DUST_I), rj2cmb(353e9, DUST_P), rj2cmb(353e9, DUST_P), 1.6, 20.)

# Constants
DUST_I = 50.
DUST_P = 10. / 1.41
mcmc_iterations = 100

# Standard Arrays
seed_list = np.arange(mcmc_iterations)
fQ_array = np.linspace(-5., 5., 20)
fQ_array[np.where(fQ_array == -1.)] = -0.9 # Replace -1. if found (causes div/0)

# Appends new parameters to modify_parameters.txt
def set_fQ_params_TCCM(fQ, fI = 1.0 , fU = 0.8):
    with open('modify_parameters.txt', 'a') as param_file:
        new_params = (rj2cmb(353e9, DUST_I/(1+fI)), rj2cmb(353e9, DUST_P/(1+fQ)), rj2cmb(353e9, DUST_P/(1+fU)), 1.6, 0.0, 20, 15.0, 1.0, fQ, 0.8)
        param_file.write(str(new_params) + "\n")
        #To input MBB model: new_params = (rj2cmb(353e9, DUST_I), rj2cmb(353e9, DUST_P), rj2cmb(353e9, DUST_P), 1.6, T)


# Construct list of fQ values and seeds to loop over
seeds, fqvals = np.meshgrid(seed_list, fQ_array)
seeds = seeds.flatten()
fqvals = fqvals.flatten()

# Loop over seeds and fQ values, assigning to MPI workers in turn
# (will do all seeds for 1 fQ value first)
for i in range(fqvals.size):
    if i % nproc != myid: continue
    print "-"*50
    print "Running %d / %d (worker %d)" % (i, fqvals.size, myid)
    print "-"*50
    mc.main(seeds[i], fqvals[i])

# Finish MPI process
comm.Barrier()
#if myid == 0: comm.Disconnect()
