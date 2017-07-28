# coding: utf-8
import numpy as np
import models
from utils import rj2cmb


DUST_I = 50.
DUST_P = 10. / 1.41

#Define parameters using modify_parameters.txt:
params_file = open('modify_parameters.txt','r')
lineList = params_file.readlines()
param_string1 = lineList[-1]
param_string2 = param_string1[1:-2] + ','
TCCM_Parameters = np.fromstring(param_string2, dtype='float', sep=',')



# "Cloud" Dust Model
# Dust same composition everywhere, but different temperature
# Magnetic field structure along the line of sight induces
# depolarization, i.e. fQ != fU

TCCM_amp_I = TCCM_Parameters[0]
TCCM_amp_Q = TCCM_Parameters[1]
TCCM_amp_U = TCCM_Parameters[2]
TCCM_beta  = TCCM_Parameters[3]
TCCM_dbeta = TCCM_Parameters[4]
TCCM_Td1   = TCCM_Parameters[5]
TCCM_Td2   = TCCM_Parameters[6]
TCCM_fI    = TCCM_Parameters[7]
TCCM_fQ    = TCCM_Parameters[8]
TCCM_fU    = TCCM_Parameters[9]

two_comp_cloud_model = models.DustGen(TCCM_amp_I, TCCM_amp_Q, TCCM_amp_U, TCCM_beta, TCCM_dbeta, TCCM_Td1, TCCM_Td2, TCCM_fI, TCCM_fQ, TCCM_fU)

#Default TCCM params:
#two_comp_cloud_model = models.DustGen(rj2cmb(353e9, DUST_I/2.), rj2cmb(353e9, DUST_P/2.1), rj2cmb(353e9, DUST_P/1.8) ,1.6,0.0,20.0,15.0,1.0,1.1,0.8)


#To implement variable MBB models:
#MBB_Parameters = np.fromstring(param_string2, dtype='float', sep=',')
# MBB_amp_I = MBB_Parameters[0]
# MBB_amp_Q = MBB_Parameters[1]
# MBB_amp_U = MBB_Parameters[2]
# MBB_beta  = MBB_Parameters[3]
# MBB_T     = MBB_Parameters[4]
#dust_model = models.DustMBB(MBB_amp_I, MBB_amp_Q, MBB_amp_U, MBB_beta, MBB_T)


dust_model = models.DustMBB(rj2cmb(353e9, DUST_I), rj2cmb(353e9, DUST_P), rj2cmb(353e9, DUST_P), 1.6, 20.)



simple_dust_model = models.DustSimpleMBB( amp_I=rj2cmb(353e9, DUST_I),
                             amp_Q=rj2cmb(353e9, DUST_P),
                             amp_U=rj2cmb(353e9, DUST_P),
                             dust_beta=1.6, dust_T=20. )

simple_dust_model_shifted = models.DustSimpleMBB(
                             amp_I=rj2cmb(353e9, DUST_I),
                             amp_Q=rj2cmb(353e9, DUST_P),
                             amp_U=rj2cmb(353e9, DUST_P),
                             dust_beta=1.7, dust_T=20. )

ame_model = models.AMEModel( amp_I=rj2cmb(30e9, 30.),
                             amp_Q=rj2cmb(30e9, 0.),
                             amp_U=rj2cmb(30e9, 0.),
                             nu_peak=25. )

ff_model = models.FreeFreeUnpol( amp_I=rj2cmb(30e9, 30.),
                                 amp_Q=rj2cmb(30e9, 0.),
                                 amp_U=rj2cmb(30e9, 0.),
                                 ff_beta=-0.118 )

sync_model = models.SyncPow( amp_I=30., amp_Q=10., amp_U=10., sync_beta=-3.2 )

cmb_model = models.CMB( amp_I=50., amp_Q=0.6, amp_U=0.6 )


# Silicate + Carbonaceous grains as 2 MBBs
two_comp_silcar_model = models.DustGen(
                                     amp_I=rj2cmb(353e9, DUST_I/1.25),
                                     amp_Q=rj2cmb(353e9, DUST_P/1.25),
                                     amp_U=rj2cmb(353e9, DUST_P/1.25),
                                     beta = 1.6,
                                     dbeta = 0.2,
                                     Td1 = 18.,
                                     Td2 = 22.,
                                     fI = 0.25,
                                     fQ = 0.25,
                                     fU = 0.25 )

# Finkbeiner 1999 Model (should perhaps update fI,fQ,FQ to
#                        representative values based on F99)
two_comp_f99_model = models.DustGen( amp_I=rj2cmb(353e9, DUST_I/1.2),
                                     amp_Q=rj2cmb(353e9, DUST_P/1.2),
                                     amp_U=rj2cmb(353e9, DUST_P/1.2),
                                     beta = 1.5,
                                     dbeta = 1.1,
                                     Td1 = 9.6,
                                     Td2 = 16.4,
                                     fI = 0.2,
                                     fQ = 0.2,
                                     fU = 0.2 )



# 2MBB Model with Fe grains
# Note: polarized orthogonally to normal dust with beta = 0
two_comp_fe_model = models.DustGen( amp_I=rj2cmb(353e9, DUST_I/1.05),
                                    amp_Q=rj2cmb(353e9, DUST_P/0.95),
                                    amp_U=rj2cmb(353e9, DUST_P/0.95),
                                    beta = 1.6,
                                    dbeta = -1.6,
                                    Td1 = 20.,
                                    Td2 = 20.,
                                    fI = 0.05,
                                    fQ = -0.05,
                                    fU = -0.05 )

# HD17 Standard Model
hd_model = models.DustHD( amp_I=rj2cmb(353e9, DUST_I),
                          amp_Q=rj2cmb(353e9, DUST_P),
                          amp_U=rj2cmb(353e9, DUST_P),
                          fcar = 1.0,
                          fsilfe = 0.,
                          uval = 0.0 )

# HD17 Model with Fe
hd_fe_model = models.DustHD( amp_I=rj2cmb(353e9, DUST_I),
                             amp_Q=rj2cmb(353e9, DUST_P),
                             amp_U=rj2cmb(353e9, DUST_P),
                             fcar = 1.e3,
                             fsilfe = 1.e3,
                             uval = 0.0 )

# General 2MBB dust model with fQ=fU
gen_dust_model = models.DustGenMBB( amp_I=rj2cmb(353e9, DUST_I),
                                    amp_Q=rj2cmb(353e9, DUST_P),
                                    amp_U=rj2cmb(353e9, DUST_P))

# Dictionary of models
model_dict = {
    'cmb':          cmb_model,
    'synch':        sync_model,
    'freefree':     ff_model,
    'mbb':          dust_model,
    'simplembb':    simple_dust_model,
    'shiftedmbb':   simple_dust_model_shifted,
    '2mbb_silcar':  two_comp_silcar_model,
    '2mbb_f99':     two_comp_f99_model,
    '2mbb_cloud':   two_comp_cloud_model,
    '2mbb_fe':      two_comp_fe_model,
    'hd':           hd_model,
    'hd_fe':        hd_fe_model,
    'genmbb':       gen_dust_model,
    'ame':          ame_model,
}
