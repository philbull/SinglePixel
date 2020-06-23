import numpy as np
import model_list, models, fitting
import copy as cp

from utils import rj2cmb

nu_pico = np.asarray([21,25,30, 36.0,43.2,51.8,62.2,74.6,89.6,
                      107.5,129.0,154.8,185.8,222.9,267.5,321.0,
                      385.2,462.2,554.7,665.6,798.7]) * 1e9

mean_beta = 1.6
mean_temp = 20.
sigma_beta=.2
sigma_temp = 4.

DUST_I = 50.
DUST_P = 5. / 1.41
amp_I=rj2cmb(353e9, DUST_I)
amp_Q=rj2cmb(353e9, DUST_P)
amp_U=rj2cmb(353e9, DUST_P)

sMBB = model_list.dust_model
#sMBB_TMF = models_list.TMFM

pMBB_narrow = model_list.prob1mbb_model_narrow
pMBB_intermediate = model_list.prob1mbb_model_intermediate
pMBB_broad = model_list.prob1mbb_model_broad
#TMFM = model_list.TMFM

TMFM_narrow = model_list.TMFM_narrow
TMFM_intermediate = model_list.TMFM_intermediate
TMFM_broad = model_list.TMFM_broad

#pMBB_smallB_bigT = models.ProbSingleMBB(amp_I=rj2cmb(353e9, DUST_I),
#                             amp_Q=rj2cmb(353e9, DUST_P),
#                             amp_U=rj2cmb(353e9, DUST_P),
#                             dust_beta=mean_beta, dust_T=mean_temp,
#                             sigma_beta=.1 * sigma_beta, sigma_temp=sigma_temp,
#                             decorrelation=False)

cmb = model_list.cmb_model
sync = model_list.sync_model

sMBB_U = models.DustMBB( amp_I=rj2cmb(353e9, DUST_I),
                             amp_Q=rj2cmb(353e9, DUST_P),
                             amp_U=rj2cmb(353e9, DUST_P),
                             dust_beta=1.5, dust_T=21. )

pMBB_U = models.ProbSingleMBB(amp_I=rj2cmb(353e9, DUST_I),
                             amp_Q=rj2cmb(353e9, DUST_P),
                             amp_U=rj2cmb(353e9, DUST_P),
                             dust_beta=1.5, dust_T=21.,
                             sigma_beta=sigma_beta, sigma_temp=sigma_temp)

TMFM_U = models.TMFM(amp_I=rj2cmb(353e9, DUST_I),
                             amp_Q=rj2cmb(353e9, DUST_P),
                             amp_U=rj2cmb(353e9, DUST_P),
                             dust_beta=1.5, dust_T=21.,
                             sigma_beta=sigma_beta,
                             mean_chi=np.pi / 4., kappa=1.0)


models_sMBB = [sMBB, cmb, sync]
#models_sMBB_TMF = [sMBB_TMF, cmb, sync]

models_pMBB_narrow = [pMBB_narrow, cmb, sync]
models_pMBB_intermediate = [pMBB_intermediate, cmb, sync]
models_pMBB_broad = [pMBB_broad, cmb, sync]

#models_TMFM_narrow = [TMFM_narrow, cmb, sync]
#models_TMFM_intermediate = [TMFM_intermediate, cmb, sync]
#models_TMFM_broad = [TMFM_broad, cmb, sync]

#sMBB_decouple = [models_sMBB, models_sMBB, [sMBB_U, cmb, sync]]
#pMBB_broad_decouple = [models_pMBB_broad, models_pMBB_broad, [pMBB_U, cmb, sync]]
#TMFM_broad_decouple = [models_TMFM_broad, models_TMFM_broad, [TMFM_U, cmb, sync]]

#models_pMBB_smallB_bigT = [pMBB_smallB_bigT, cmb, sync]

def make_pnames(models_fit):
    amp_names = []
    param_names = []

    for mod in models_fit:
        # Parameter names
        amp_names += ["%s_%s" % (mod.model, pol) for pol in "IQU"]
        param_names += mod.param_names

    return amp_names + param_names

pnames_sMBB = make_pnames(models_sMBB)
#pnames_pMBB_narrow = make_pnames(models_pMBB_narrow)
#pnames_pMBB_intermediate = make_pnames(models_pMBB_intermediate)
pnames_pMBB = make_pnames(models_pMBB_broad)

pnames_sMBB, pnames_pMBB

def make_signal(components):
    signal = 0

    for comp in components:
        # Add this component to total signal
        signal += np.atleast_2d(comp.amps()).T * comp.scaling(nu_pico)

        # Store CMB signal separately
        if comp.model == 'cmb':
            cmb_signal = np.atleast_2d(comp.amps()).T * comp.scaling(nu_pico)

    return signal

fsigma_T=1e5
fsigma_P=1.

beam_mat = np.identity(3*len(nu_pico)) # Beam model

# pvals set the model parameters
params_sMBB = [sMBB.amp_I, sMBB.amp_Q, sMBB.amp_U,
               cmb.amp_I, cmb.amp_Q, cmb.amp_U,
               sync.amp_I, sync.amp_Q, sync.amp_U,
               sMBB.dust_beta, sMBB.dust_T,
               sync.sync_beta]
params_pMBB_narrow = [pMBB_narrow.amp_I, pMBB_narrow.amp_Q, pMBB_narrow.amp_U,
                      cmb.amp_I, cmb.amp_Q, cmb.amp_U,
                      sync.amp_I, sync.amp_Q, sync.amp_U,
                      pMBB_narrow.dust_beta, pMBB_narrow.dust_T,
                      pMBB_narrow.sigma_beta, pMBB_narrow.sigma_temp, sync.sync_beta]
params_pMBB_intermediate = [pMBB_intermediate.amp_I, pMBB_intermediate.amp_Q, pMBB_intermediate.amp_U,
                            cmb.amp_I, cmb.amp_Q, cmb.amp_U,
                            sync.amp_I, sync.amp_Q, sync.amp_U,
                            pMBB_intermediate.dust_beta, pMBB_intermediate.dust_T,
                            pMBB_intermediate.sigma_beta, pMBB_intermediate.sigma_temp, sync.sync_beta]
params_pMBB_broad = [pMBB_broad.amp_I, pMBB_broad.amp_Q, pMBB_broad.amp_U,
                     cmb.amp_I, cmb.amp_Q, cmb.amp_U,
                     sync.amp_I, sync.amp_Q, sync.amp_U,
                     pMBB_broad.dust_beta, pMBB_narrow.dust_T,
                     pMBB_broad.sigma_beta, pMBB_broad.sigma_temp, sync.sync_beta]

initial_vals_sMBB2pMBB = cp.deepcopy([pMBB_narrow.amp_I, pMBB_narrow.amp_Q, pMBB_narrow.amp_U,
                      cmb.amp_I, cmb.amp_Q, cmb.amp_U,
                      sync.amp_I, sync.amp_Q, sync.amp_U,
                      pMBB_narrow.dust_beta, pMBB_narrow.dust_T,
                      0, 0, sync.sync_beta]) #(amp_I, amp_Q, amp_U, cmb.amp_I, cmb.amp_Q, cmb.amp_U, sync.amp_I, sync.amp_Q, sync.amp_U, mean_beta, mean_temp, 1e-2, 5e-3, sync.sync_beta)

initial_vals_sMBB_decouple = (sMBB.amp_I, sMBB.amp_Q, sMBB.amp_U, cmb.amp_I, cmb.amp_Q, cmb.amp_U,
              sync.amp_I, sync.amp_Q, sync.amp_U, sMBB.dust_beta, sMBB.dust_T,
              sync.sync_beta, sMBB.dust_beta, sMBB.dust_T,
              sync.sync_beta, sMBB_U.dust_beta, sMBB_U.dust_T,
              sync.sync_beta)

initial_vals_pMBB_broad = cp.deepcopy(params_pMBB_broad)

def single_fit(models_data, models_fit, initial_vals, nu=nu_pico, decouple=False, noiseless=False):
    parent_model = 'mbb'
    D_vec, Ninv = fitting.generate_data(nu, fsigma_T, fsigma_P, models_data,
                                            noise_file="data/noise_pico.dat",
                                            decouple=decouple)
    print('initial vals: ', initial_vals)
    if noiseless is True:
        D_vec = np.mat(make_signal(models_data).reshape(63,1))


    data_spec = (nu_pico, D_vec, Ninv, beam_mat)
    p_spec = (pnames_pMBB, initial_vals, parent_model)

    pnames_out, samples, logp  = fitting.joint_mcmc(data_spec,
                                                models_fit, p_spec,
                                                nwalkers=48, burn=1000,
                                                steps=10000, nthreads=1,
                                                sample_file=None, decouple=decouple)

    return pnames_out, samples, logp

## sMBB data

#sMBB_single = single_fit(models_sMBB, models_pMBB_narrow, initial_vals_sMBB2pMBB)

#pMBB_narrow_single = single_fit(models_pMBB_narrow, models_pMBB_narrow, initial_vals_pMBB_broad)
#pMBB_intermediate_single = single_fit(models_pMBB_intermediate, models_pMBB_narrow, initial_vals_pMBB_broad)
pMBB_broad_single = single_fit(models_pMBB_broad, models_pMBB_broad, initial_vals_pMBB_broad, noiseless=True)

#np.save('sMBB_single_trueinitial', sMBB_single)

#np.save('pMBB_narrow_single', pMBB_narrow_single)
#np.save('pMBB_intermediate_single', pMBB_intermediate_single
np.save('pMBB_broad_single_noiseless', pMBB_broad_single)

#sMBB_single_decouple = single_fit(sMBB_decouple, models_sMBB,
#                                    initial_vals_sMBB_decouple, decouple=True)
