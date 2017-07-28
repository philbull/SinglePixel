import numpy as np
import matplotlib.pyplot as plt



a = np.loadtxt('convergence_of_mean_I_biasv0.txt')
plt.plot(a)
plt.title("Fitting TCCM to GenMBB" + "\n" + "Default Parameters, Except fQ = -5" + "\nNSTEPS = 5000, BURN = 500, NWALKERS = 100")
plt.ylabel("Mean I Bias")
plt.xlabel("# Of MCMC Iterations")
plt.show()
