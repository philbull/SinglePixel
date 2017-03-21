#!/usr/bin/python
"""
Example model SEDs
"""
import numpy as np
import models
from utils import rj2cmb, cmb2rj, bands_log
from model_list import *
import pylab as P

# Reference noise curve
NOISE_FILE = "data/core_plus_extended_noise.dat"

# Define input models and their amplitudes/parameters

# Dictionary of models
model_dict = {
    'cmb':          cmb_model, 
    'synch':        sync_model, 
    #'freefree':     ff_model,
    'mbb':          dust_model, 
    #'simplembb':    simple_dust_model,
    #'shiftedmbb':   simple_dust_model_shifted,
    '2mbb_silcar':  two_comp_silcar_model,
    '2mbb_f99':     two_comp_f99_model,
    '2mbb_cloud':   two_comp_cloud_model,
    '2mbb_fe':      two_comp_fe_model,
    'hd':           hd_model,
    'hd_fe':        hd_fe_model,
    #'ame':          ame_model,
}

colour_dict = {
    'cmb':          'k',
    'synch':        '#3954EC', 
    'mbb':          '#AF1934',
    '2mbb_f99':     '#9FD69B',
    '2mbb_silcar':  '#2EA822',
    '2mbb_cloud':   '#FFBFC7',
    '2mbb_fe':      '#EC1E1C',
    'hd':           '#FFDD45',
    'hd_fe':        '#ECA61C',
    'ame':          '#F6B7E1', #'#D60591',
}

dash_dict = {
    'cmb':          [],
    'synch':        [], 
    'mbb':          [8,3], 
    '2mbb_f99':     [],
    '2mbb_silcar':  [5,3,2,3],
    '2mbb_cloud':   [],
    '2mbb_fe':      [5,3],
    'hd':           [],
    'hd_fe':        [5,3],
    'ame':          [],
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
axx = P.subplot(223)

# Loop over models and plot them
for mname in model_dict.keys():
    mod = model_dict[mname]
    I, Q, U = np.atleast_2d(mod.amps()).T * mod.scaling(nu)
    
    ax1.plot(nu/1e9, I, lw=1.8, label=mname, color=colour_dict[mname], 
             dashes=dash_dict[mname])
    ax2.plot(nu/1e9, np.abs(Q), lw=1.8, color=colour_dict[mname], 
             dashes=dash_dict[mname])
    ax3.plot(nu/1e9, np.abs(U), lw=1.8, color=colour_dict[mname], 
             dashes=dash_dict[mname])
    
    axx.plot(nu/1e9, cmb2rj(nu, I), lw=1.8, color=colour_dict[mname], 
             dashes=dash_dict[mname])


#ax1.plot(353., rj2cmb(353e9, 150.), 'kx') # I dust amp.
#ax2.plot(353., rj2cmb(353e9, 10.), 'kx') # Q
#ax3.plot(353., rj2cmb(353e9, 10.), 'kx') # U
#axx.plot(353., 150., 'kx')

ax1.legend(loc='upper center', frameon=False, ncol=2, fontsize=14)

ax1.set_ylabel(r"$\Delta T$ $[\mu{\rm K}_{\rm CMB}]$", fontsize=18)
ax2.set_ylabel(r"$\Delta Q$ $[\mu{\rm K}_{\rm CMB}]$", fontsize=18)
ax3.set_ylabel(r"$\Delta U$ $[\mu{\rm K}_{\rm CMB}]$", fontsize=18)
axx.set_ylabel(r"$\Delta T$ $[\mu{\rm K}_{\rm RJ}]$", fontsize=18)

for ax in [ax1, ax2, ax3]:
    ax.set_xlabel(r"$\nu$ $[{\rm GHz}]$", fontsize=18)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim((1., 800.))
    ax.set_ylim((1e-2, 1e8))

# FIXME
axx.set_xlabel(r"$\nu$ $[{\rm GHz}]$", fontsize=18)
axx.set_xscale('log')
axx.set_yscale('log')
axx.set_xlim((1., 800.))
axx.set_ylim((1e-2, 1e3))

P.tight_layout()
P.show()

