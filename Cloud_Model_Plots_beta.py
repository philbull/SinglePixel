
import matplotlib.pyplot as plt
import numpy as np
import models
from utils import cmb2rj, rj2cmb
#Stokes Variables: 0 = I, 1 = Q, 2 = U
#amp_I about 322.6, amp_Q about 43.58, amp_U about 50.85
#Default Cloud Model Parameters: beta = 1.6, dbeta = 0, Td1 = 20,Td2 = 15, fI = 1.0, fQ = 1.1, fU = 0.8

DUST_I = 50.
DUST_P = (3.5 * 1.41) / 1.41
PRECISION = 400
#Sets up plot axes
nu_Hz = np.logspace(np.log10(1e9), np.log10(800e9), PRECISION)
nu_GHz = nu_Hz / 1.0e9

def cloud_model_stokes(frequencies, stokes_var, beta_ , d_beta, T1, T2, f_I, f_Q, f_U):
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
    #Verify these amplification factors with Brandon!!

    if stokes_var == 0:
        amp_factor = rj2cmb(353e9, DUST_I/(1+f_I))
    if stokes_var == 1:
        amp_factor = rj2cmb(353e9, DUST_P/(1+f_Q))
    if stokes_var == 2:
        amp_factor = rj2cmb(353e9, DUST_P/(1+f_U))

    return two_comp_cloud_model.scaling(frequencies)[stokes_var]*amp_factor

def y_label(stokes_var):
    if stokes_var == 0:
        y_axis_label = "Intensity ($\mu$K$_{RJ}$)"
    if stokes_var == 1:
        y_axis_label = "Intensity of Q Polarization($\mu$K$_{RJ}$)"
    if stokes_var == 2:
        y_axis_label = "Intensity of U Polarization($\mu$K$_{RJ}$)"
    return y_axis_label



def cmb2rj_convert_and_scale(frequencies, T_cmb_intensities, stokes_var):  #stokes_var may not be necessary once
    return np.array(cmb2rj(frequencies, T_cmb_intensities))             #scaling is fixed.


# def produce_rj_plot(frequencies, stokes_var, beta, d_beta, T1, T2, fI, fQ, fU):
#     #Labels are Focused on changing D_Beta   Conveient: $\Delta$B
#     label_string = round(T1, 2)
#     plt.plot(nu_GHz, cmb2rj_convert_and_scale(frequencies, cloud_model_stokes(frequencies,stokes_var,beta,d_beta,T1,T2,fI,fQ,fU), 0), label = "T1 = " + str(label_string)) #adding stokes_var just in case its needed later


#Parameters are from a list [beta, d_Beta, T1, T2, fI, fQ, fU]
def vary_params(frequencies, stokes_var, varying_parameter_position, other_parameters, start_value, finish_value, step):
    parameters = other_parameters
    parameters.insert(varying_parameter_position, 0)

    list_of_strings = ["B", "$\Delta$B", "T$_{1}$", "T$_{2}$", "f$_{I}$",  "f$_{Q}$", "f$_{U}$"]
    label_string = list_of_strings[varying_parameter_position]

    title_phrase = ("Varying " + label_string + " from " + str(start_value) + " to " + str(finish_value) +
                    " Using Steps of "+ str(step) + "\n")

    list_of_strings.remove(label_string)

    for i in range(len(list_of_strings)):
         title_phrase += ("     " + list_of_strings[i] + " = " +  str(other_parameters[i]))

    #beta, d_Beta, T1, T2, fI, fQ, fU
    #roundabout way to get other param values in title

    plt.title("Two Cloud Model" + "\n" +  title_phrase + "\n" + str(PRECISION) + " Frequency Samples")

    varying_param_steps = np.arange(start_value, finish_value + step,step)



    plt.title("Two Cloud Model" + "\n" +  title_phrase + "\n" + str(PRECISION) + " Frequency Samples")

    plt.subplot(2,1,1)
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel(y_label(stokes_var))
    plt.xlabel(" $\\nu$ (GHz)")
    plt.axis([min(nu_GHz), max(nu_GHz), 1e-5, 1e3])

    for i in varying_param_steps:

        stokes_vs_frequency = cmb2rj_convert_and_scale(frequencies, cloud_model_stokes(frequencies,stokes_var,
                                parameters[0],parameters[1],parameters[2],parameters[3],
                                parameters[4],parameters[5],parameters[6]), stokes_var)
        parameters[varying_parameter_position] = i

        plt.plot(nu_GHz, stokes_vs_frequency , label = label_string + " = "+ str(i)) #adding stokes_var just in case its needed later


    plt.subplot(2,1,2)
    #plt.yscale('log')
    #plt.xscale('log')
    plt.ylabel("Polarization Angle (deg)")
    plt.xlabel(" $\\nu$ (GHz)")
    plt.axis([min(nu_GHz), max(nu_GHz), 0, 40])

    for i in varying_param_steps:
        Q = cmb2rj_convert_and_scale(frequencies, cloud_model_stokes(frequencies, 1,parameters[0],parameters[1],parameters[2],parameters[3],parameters[4],parameters[5],parameters[6]), 1)
        U = cmb2rj_convert_and_scale(frequencies, cloud_model_stokes(frequencies, 2,parameters[0],parameters[1],parameters[2],parameters[3],parameters[4],parameters[5],parameters[6]), 2)

        angle_values = np.degrees(0.5*np.arctan2(Q,U))
        plt.plot(nu_GHz, angle_values)




    # if varying_parameter_position == 5:
    #     plt.xlabel("f$_{Q}$")
    #     fU = other_parameters[-1]
    #     angle_values = np.degrees([0.5*np.arctan2(fU,fQ) for fQ in varying_param_steps])
    # if varying_parameter_position == 6:
    #     plt.xlabel("f$_{U}$")
    #     fQ = other_parameters[-1]
    #     angle_values = np.degrees([0.5*np.arctan2(fU,fQ) for fU in varying_param_steps])
    #
    # plt.plot(varying_param_steps, angle_values)
    # plt.ylabel("$\\theta$ (Polarization Angle in Degrees)")
    # plt.axis([min(varying_param_steps), max(varying_param_steps), min(angle_values), max(angle_values)])

#--------------------Main Program--------------------------------------#
#Takes inputs (frequencies, stokes_var, varying_parameter_position, [other_parameters], start_value, finish_value, step)
#List of parameters includes[beta, d_Beta, T1, T2, fI, fQ, fU]]
vary_params(nu_Hz, 0, 3, [1.6, 5.0, 20.0, 1.0, 1.0 , 1.0], 1, 5, 0.1 )



#Plot Setup
plt.grid("on")

plt.legend(loc = "lower right")

plt.show()
