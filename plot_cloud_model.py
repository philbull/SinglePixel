import emcee
import matplotlib.pyplot as plt
import numpy as np
import models
from utils import cmb2rj, rj2cmb

#Stokes Variables: 0 = I, 1 = Q, 2 = U
#amp_I about 322.6, amp_Q about 43.58, amp_U about 50.85
#Default Cloud Model Parameters: beta = 1.6, dbeta = 0, Td1 = 20,Td2 = 15, fI = 1.0, fQ = 1.1, fU = 0.8

DUST_I = 50.
DUST_P = 3.50
PRECISION = 250
#Sets up plot axes
nu_Hz = np.logspace(np.log10(10e9), np.log10(1000e9), PRECISION)
nu_GHz = nu_Hz / 1.0e9

def wrapper(fQ_list):
    def cloud_model_stokes_CMB(frequencies, stokes_var, beta_ , d_beta, T1, T2, f_I, f_Q, f_U):
        two_comp_cloud_model = models.DustGen(
                                            amp_I=rj2cmb(353e9, DUST_I/(1+f_I)),
                                            amp_Q=rj2cmb(353e9, DUST_P/(1+f_Q)),
                                            amp_U=rj2cmb(353e9, DUST_P/(1+f_U)),
                                            beta = beta_,
                                            dbeta = d_beta,
                                            Td1 = T1,
                                            Td2 = T2,
                                            fI = f_I,
                                            fQ = f_Q,
                                            fU = f_U )
        if stokes_var == 0:
            amp_factor = rj2cmb(353e9, DUST_I/(1+f_I))
        elif stokes_var == 1:
            amp_factor = rj2cmb(353e9, DUST_P/(1+f_Q))
        elif stokes_var == 2:
            amp_factor = rj2cmb(353e9, DUST_P/(1+f_U))

        return two_comp_cloud_model.scaling(frequencies)[stokes_var] * amp_factor

    def y_label(stokes_var):
        if stokes_var == 0:
            y_axis_label = "Intensity ($\mu$K$_{RJ}$)"
        if stokes_var == 1:
            y_axis_label = "Q Polarization($\mu$K$_{RJ}$)"
        if stokes_var == 2:
            y_axis_label = "U Polarization($\mu$K$_{RJ}$)"
        return y_axis_label

    def plot_labels(parameter, value):
        if parameter == "b":
            labelfrag = "B"
        if parameter == "db":
            labelfrag = "$\Delta$B"
        if parameter == "T1":
            labelfrag = "T$_{1}$"
        if parameter == "T2":
            labelfrag = "T$_{2}$"
        if parameter == "fI":
            labelfrag = "f$_{I}$"
        if parameter == "fQ":
            labelfrag = "f$_{Q}$"
        if parameter == "fU":
            labelfrag = "f$_{U}}$"
        return(labelfrag + " = " + str(round(value,2)))

    def rj_stokes_data(frequencies, stokes_var, beta, d_beta, T1, T2, fI, fQ, fU):
        return(cmb2rj(frequencies, cloud_model_stokes_CMB(frequencies,stokes_var,beta,d_beta,T1,T2,fI,fQ,fU)))


    #Plots angles corresponding to permuation of cloud parameters based on V
    def angle_plot(frequencies, beta, d_beta, T1, T2, fI, fQ, fU):
        Q = rj_stokes_data(frequencies, 1, beta, d_beta, T1, T2, fI, fQ, fU)
        U = rj_stokes_data(frequencies, 2, beta, d_beta, T1, T2, fI, fQ, fU)
        angle_values = 0.5*np.arctan2(U,Q)
        plt.plot(nu_GHz, np.degrees(angle_values))


    def modification_of_fQ(list_of_fQ_values):

        for i in list_of_fQ_values:

            plt.subplot(332)
            plt.ylabel(y_label(0))
            plt.xlabel(" $\\nu$ (GHz)")
            plt.yscale('log')
            plt.xscale('log')
            plt.grid("on")
            plt.plot(nu_GHz, rj_stokes_data(nu_Hz, 0, 1.6, 0, 20, 15, 1, i, 1))


            plt.subplot(335)
            plt.ylabel(y_label(1))
            plt.xlabel(" $\\nu$ (GHz)")
            plt.yscale('log')
            plt.xscale('log')
            plt.grid("on")
            plt.plot(nu_GHz, rj_stokes_data(nu_Hz, 1, 1.6, 0, 20, 15, 1, i, 1))



            plt.subplot(338)
            plt.ylabel(y_label(2))
            plt.xlabel(" $\\nu$ (GHz)")
            plt.yscale('log')
            plt.xscale('log')
            plt.grid("on")
            plt.plot(nu_GHz, rj_stokes_data(nu_Hz, 2, 1.6, 0, 20, 15, 1, i, 1))


            plt.subplot(336)
            plt.ylabel("Polarization Angle (deg)")
            plt.xlabel(" $\\nu$ (GHz)")
            plt.grid("on")
            plt.axis([min(nu_GHz), max(nu_GHz), 20, 27])
            angle_plot(nu_Hz, 1.6, 0, 20, 15, 1, i, 1)

    modification_of_fQ(fQ_list)
