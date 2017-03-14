#!/usr/bin/python
"""
Example model SEDs
"""
import numpy as np
import models
from utils import rj2cmb, bands_log
import pylab as P

# Reference noise curve
NOISE_FILE = "data/core_plus_extended_noise.dat"

# Define input models and their amplitudes/parameters
dust_model = models.DustMBB( amp_I=rj2cmb(353e9, 150.), 
                             amp_Q=rj2cmb(353e9, 10.), 
                             amp_U=rj2cmb(353e9, 10.), 
                             dust_beta=1.6, dust_T=20. )
simple_dust_model = models.DustSimpleMBB( amp_I=rj2cmb(353e9, 150.), 
                             amp_Q=rj2cmb(353e9, 10.), 
                             amp_U=rj2cmb(353e9, 10.), 
                             dust_beta=1.6, dust_T=20. )
simple_dust_model_shifted = models.DustSimpleMBB( 
                             amp_I=rj2cmb(353e9, 150.), 
                             amp_Q=rj2cmb(353e9, 10.), 
                             amp_U=rj2cmb(353e9, 10.), 
                             dust_beta=1.7, dust_T=20. )
sync_model = models.SyncPow( amp_I=30., amp_Q=10., amp_U=10., sync_beta=-3.2 )
cmb_model = models.CMB( amp_I=50., amp_Q=0.6, amp_U=0.6 )


# Silicate + Carbonaceous grains as 2 MBBs
two_comp_silcar_model = models.DustGen( amp_I=rj2cmb(353e9, 150.),
                                     amp_Q=rj2cmb(353e9, 10.),
                                     amp_U=rj2cmb(353e9, 10.),
                                     beta = 1.6,
                                     dbeta = 0.2,
                                     Td1 = 18.,
                                     Td2 = 22.,
                                     fI = 0.25,
                                     fQ = 0.25,
                                     fU = 0.25)

# Finkbeiner 1999 Model (should perhaps update fI,fQ,FQ to
#                        representative values based on F99)
two_comp_f99_model = models.DustGen( amp_I=rj2cmb(353e9, 150.),
                                     amp_Q=rj2cmb(353e9, 10.),
                                     amp_U=rj2cmb(353e9, 10.),
                                     beta = 1.5,
                                     dbeta = 1.1,
                                     Td1 = 9.6,
                                     Td2 = 16.4,
                                     fI = 0.2,
                                     fQ = 0.2,
                                     fU = 0.2)

# "Cloud" Dust Model
# Dust same composition everywhere, but different temperature
# Magnetic field structure along the line of sight induces
# depolarization, i.e. fQ != fU
two_comp_cloud_model = models.DustGen( amp_I=rj2cmb(353e9, 150.),
                                     amp_Q=rj2cmb(353e9, 10.),
                                     amp_U=rj2cmb(353e9, 10.),
                                     beta = 1.6,
                                     dbeta = 0.,
                                     Td1 = 20.,
                                     Td2 = 15.,
                                     fI = 1.0,
                                     fQ = 1.1,
                                     fU = 0.8)

# 2MBB Model with Fe grains
# Note: polarized orthogonally to normal dust with beta = 0
two_comp_fe_model = models.DustGen( amp_I=rj2cmb(353e9, 150.),
                                     amp_Q=rj2cmb(353e9, 10.),
                                     amp_U=rj2cmb(353e9, 10.),
                                     beta = 1.6,
                                     dbeta = -1.6,
                                     Td1 = 20.,
                                     Td2 = 20.,
                                     fI = 0.05,
                                     fQ = -0.05,
                                     fU = -0.05)

# HD17 Standard Model
hd_model = models.DustHD( amp_I=rj2cmb(353e9, 150.),
                                     amp_Q=rj2cmb(353e9, 10.),
                                     amp_U=rj2cmb(353e9, 10.),
                                     fcar = 1.0,
                                     fsilfe = 0.,
                                     uval = 0.0)

# HD17 Model with Fe
hd_fe_model = models.DustHD( amp_I=rj2cmb(353e9, 150.),
                                     amp_Q=rj2cmb(353e9, 10.),
                                     amp_U=rj2cmb(353e9, 10.),
                                     fcar = 1.e3,
                                     fsilfe = 1.e3,
                                     uval = 0.0)

# Dictionary of models
model_dict = {
    'cmb':        cmb_model, 
    'synch':      sync_model, 
    'mbb':        dust_model, 
    'simplembb':  simple_dust_model,
    'shiftedmbb': simple_dust_model_shifted,
    '2mmb_silcar': two_comp_silcar_model,
    '2mmb_f99': two_comp_f99_model,
    '2mbb_cloud': two_comp_cloud_model,
    '2mbb_fe': two_comp_fe_model,
    'hd': hd_model,
    'hd_fe': hd_fe_model
}

in1 = ['synch', 'simplembb']
fit1 = ['synch', 'simplembb']

in2 = ['synch', '2mmb_silcar']
in3 = ['synch', '2mbb_f99']
in4 = ['synch', '2mbb_cloud']
in5 = ['synch', '2mbb_fe']
in6 = ['synch', 'hd']
in7 = ['synch', 'hd_fe']


# Frequency array
nu = np.logspace(0., np.log10(800.), 300) * 1e9 # Freq. in Hz

ax1 = P.subplot(221)
ax2 = P.subplot(222)
ax3 = P.subplot(224)

# Loop over models and plot them
for mname in model_dict.keys():
    mod = model_dict[mname]
    I, Q, U = np.atleast_2d(mod.amps()).T * mod.scaling(nu)
    
    ax1.plot(nu/1e9, I, lw=1.8, label=mname)
    ax2.plot(nu/1e9, np.abs(Q), lw=1.8)
    ax3.plot(nu/1e9, np.abs(U), lw=1.8)

ax1.legend(loc='upper center', frameon=False, ncol=2, fontsize=12)

ax1.set_xlabel(r"$\nu$ $[(\rm GHz)]$", fontsize=18)
ax1.set_ylabel(r"$\Delta T$ $[\mu{\rm K}]$", fontsize=18)

for ax in [ax1, ax2, ax3]:
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim((1., 800.))
    ax.set_ylim((1e-2, 1e8))

P.tight_layout()
P.show()

