
import models
from utils import rj2cmb

DUST_I = 50.
DUST_P = 5. / 1.41

# Define input models and their amplitudes/parameters
dust_model = models.DustMBB( amp_I=rj2cmb(353e9, DUST_I),
                             amp_Q=rj2cmb(353e9, DUST_P),
                             amp_U=rj2cmb(353e9, DUST_P),
                             dust_beta=1.6, dust_T=20. )

prob1mbb_model = models.ProbSingleMBB( amp_I=rj2cmb(353e9, DUST_I),
                             amp_Q=rj2cmb(353e9, DUST_P),
                             amp_U=rj2cmb(353e9, DUST_P),
                             dust_beta=1.6, dust_T=20.,
                             sigma_beta=.2, sigma_temp=4. )

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

sync_model = models.SyncPow( amp_I=rj2cmb(30e9, 30.),
                             amp_Q=rj2cmb(30e9, 10.),
                             amp_U=rj2cmb(30e9, 10.),
                             sync_beta=-1.2 )

cmb_model = models.CMB( amp_I=50., amp_Q=0.6, amp_U=0.6 )


# Silicate + Carbonaceous grains as 2 MBBs
two_comp_silcar_model = models.DustGen(
                                     amp_I=rj2cmb(353e9, DUST_I/1.25),
                                     amp_Q=rj2cmb(353e9, DUST_P),
                                     amp_U=rj2cmb(353e9, DUST_P),
                                     beta = 1.6,
                                     dbeta = 0.2,
                                     Td1 = 15.,
                                     Td2 = 24.,
                                     fI = 0.25,
                                     fQ = 0.,
                                     fU = 0. )

# Meisner+Finkbeiner 2015 model
two_comp_f99_model = models.DustGen( amp_I=rj2cmb(353e9, DUST_I/1.19),
                                     amp_Q=rj2cmb(353e9, DUST_P/1.19),
                                     amp_U=rj2cmb(353e9, DUST_P/1.19),
                                     beta = 1.63,
                                     dbeta = 1.19,
                                     Td1 = 9.75,
                                     Td2 = 15.7,
                                     fI = 0.19,
                                     fQ = 0.19,
                                     fU = 0.19 )

# "Cloud" Dust Model
# Dust same composition everywhere, but different temperature
# Magnetic field structure along the line of sight induces
# depolarization, i.e. fQ != fU
two_comp_cloud_model = models.DustGen(
                                     amp_I=rj2cmb(353e9, DUST_I/2.),
                                     amp_Q=rj2cmb(353e9, DUST_P/3.0),
                                     amp_U=rj2cmb(353e9, DUST_P/1.5),
                                     beta = 1.6,
                                     dbeta = 0.,
                                     Td1 = 20.,
                                     Td2 = 15.,
                                     fI = 1.0,
                                     fQ = 2.0,
                                     fU = 0.5 )

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
                                    amp_U=rj2cmb(353e9, DUST_P) )

# General 2MBB dust model with fQ!=fU
gen_dust_model_depol = models.DustGenMBBDepol( amp_I=rj2cmb(353e9, DUST_I),
                                               amp_Q=rj2cmb(353e9, DUST_P),
                                               amp_U=rj2cmb(353e9, DUST_P) )

# 2MBB dust model with parameters set exactly to the MF15 values
gen_dust_model_mf15 = models.DustGenMBB( amp_I=rj2cmb(353e9, DUST_I/1.19),
                                         amp_Q=rj2cmb(353e9, DUST_P/1.19),
                                         amp_U=rj2cmb(353e9, DUST_P/1.19),
                                         beta = 1.63,
                                         dbeta = 1.19,
                                         Td1 = 9.75,
                                         Td2 = 15.7,
                                         fI = 0.19,
                                         fQ = 0.19 )

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
    '2mbb_mf15':    two_comp_f99_model,
    '2mbb_cloud':   two_comp_cloud_model,
    '2mbb_fe':      two_comp_fe_model,
    'hd':           hd_model,
    'hd_fe':        hd_fe_model,
    'genmbb':       gen_dust_model,
    'genmbbdp':     gen_dust_model_depol,
    'genmbb15':     gen_dust_model_mf15,
    'ame':          ame_model,
}
