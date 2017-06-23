import matplotlib.pyplot as plt
import numpy as np
import models
from utils import rj2cmb

DUST_I = 50.
DUST_P = 10. / 1.41
amount_of_curves_summed = 5000              #The MBB dust model is executed this number of times, then averaged artithmetically to produce one line.
PRECISION = 300                            #Number of data points per line.
amount_of_sums = 5                        #Number of lines ultimately displayed in the plot. Essentially the number of trials.
nu = np.linspace(50, 1000, PRECISION)*1e9    #Generates a list of frequencies for the MBB model to test, in GHz.
dust_beta_placeholder = 0
dust_T_placeholder = 0
cmb_model = models.CMB( amp_I=50., amp_Q=0.6, amp_U=0.6 )




#Generates the curves plotted.

for i in range(0, amount_of_sums):                   #This outer loop allows us to plot multiple lines at a time.
    aggregated_curves = np.zeros(PRECISION)
    for i in range(0, amount_of_curves_summed):      #In each cycle of this inner loop, a pair of B,T values are randomly generated, and alters the dust model.

        dust_beta_placeholder = np.random.uniform(1.0, 2.4)
        dust_T_placeholder = np.random.uniform(10,30)
        dust_model = models.DustMBB( amp_I=rj2cmb(353e9, DUST_I), #Imported from model_list.py
                                     amp_Q=rj2cmb(353e9, DUST_P),
                                     amp_U=rj2cmb(353e9, DUST_P),
                                     dust_beta= dust_T_placeholder, dust_T = dust_T_placeholder)
        aggregated_curves  = aggregated_curves + np.array(dust_model.scaling(nu)[0])
    plt.plot(nu, aggregated_curves/amount_of_curves_summed)


plt.plot(nu, cmb_model.scaling(nu)[0])                 #This line plots the CMB.


#Formats plot
plt.yscale('log')
plt.title(str(amount_of_sums) + " Composite Spectra of " + str(amount_of_curves_summed) + " Averaged Curves Each" + "\n" + "Precision: " + str(PRECISION) + " Frequency Values")
plt.xlabel("Frequency (Hz)")
plt.ylabel("T (K)")
plt.show()
