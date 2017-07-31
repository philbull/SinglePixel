#FIXME: NEEds work
import model_list_experimental
from utils import *
import numpy as np
import matplotlib.pyplot as plt
import models

DUST_I = 50.
DUST_P = 10. / 1.41

wavelengths = np.logspace(np.log10(10e9), np.log10(1000e9))
plot_axis = wavelengths / 1e9


#sync_scaling_factor = 30 / cmb2rj(30e9, model_list_experimental.sync_model.scaling(30e9)[0]) #Applies figure frequencies from Table 1
sync_scaling_factor = 1



plt.figure(figsize = (7,5))
plt.suptitle("Synchrotron SEDs")

plt.subplot(121)
#plt.plot(plot_axis, cmb2rj(wavelengths, model_list_experimental.cmb_model.scaling(wavelengths)[0]))
plt.plot(plot_axis, sync_scaling_factor * cmb2rj(wavelengths, model_list_experimental.sync_model.scaling(wavelengths)[0]))
#plt.plot(plot_axis, ff_scaling_factor * cmb2rj(wavelengths, model_list_experimental.ff_model.scaling(wavelengths)[0]))
#plt.plot(plot_axis, ame_scaling_factor * cmb2rj(wavelengths, model_list_experimental.ame_model.scaling(wavelengths)[0]))
#plt.plot(plot_axis, dust_scaling_factor * cmb2rj(wavelengths, dust_model.scaling(wavelengths)[0]))

plt.xscale("log")
plt.yscale("log")
plt.xlabel("$\\nu$ (GHz)")
plt.ylabel('I ($\\mu$$K_{RJ}$)')
plt.ylim(1e-3, 1e4)


plt.subplot(122)
#plt.plot(plot_axis, cmb2rj(wavelengths, model_list_experimental.cmb_model.scaling(wavelengths)[1]))
plt.plot(plot_axis, sync_scaling_factor * cmb2rj(wavelengths, model_list_experimental.sync_model.scaling(wavelengths)[1]))
#plt.plot(plot_axis, dust_scaling_factor * cmb2rj(wavelengths, model_list_experimental.dust_model.scaling(wavelengths)[1]))

plt.xscale("log")
plt.yscale("log")
plt.xlabel("$\\nu$ (GHz)")
plt.ylabel('Q ($\\mu$$K_{RJ}$)')


plt.tight_layout()
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.85, wspace=0.5, hspace=0.5)
plt.legend(loc = "best")
plt.show()
