import numpy as np
import matplotlib.pyplot as plt
import cut_bias as bias
#from playsound import playsound

cuts = np.arange(0,10000,200)

#Assuming seed == 0
I_bias = []
Q_bias = []
U_bias = []

for i in cuts:
    parameter_biases = bias.bias_array(i)
    I_bias.append(parameter_biases[0])
    Q_bias.append(parameter_biases[1])
    U_bias.append(parameter_biases[2])


# playsound("finish.mp3")
# playsound("sound.mp3")

plt.suptitle("Dependence of CMB Bias Generated in a Single MCMC vs Steps Cut" +
             "\nIn: TCCM Fit: MBB | Default Parameters (except fQ = -5)" +
             "\nNSTEPS = 100000, BURN = 10" )

plt.subplot(221)
plt.plot(cuts, I_bias)
plt.xlabel("Cut")
plt.ylabel("I Bias")
plt.grid("on")

plt.subplot(222)
plt.plot(cuts, Q_bias)
plt.xlabel("Cut")
plt.ylabel("Q Bias")
plt.grid("on")

plt.subplot(223)
plt.plot(cuts, U_bias)
plt.xlabel("Cut")
plt.ylabel("U Bias")
plt.grid("on")

plt.tight_layout()
plt.subplots_adjust(top=0.75)
plt.show()
