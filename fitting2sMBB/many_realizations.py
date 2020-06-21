import numpy as np
import model_list, models, fitting
import copy as cp

from utils import rj2cmb

nu_pico = np.asarray([21,25,30, 36.0,43.2,51.8,62.2,74.6,89.6,
                      107.5,129.0,154.8,185.8,222.9,267.5,321.0,
                      385.2,462.2,554.7,665.6,798.7]) * 1e9

mean_beta = 1.6
mean_temp = 20.
sigma_beta = .2
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
TMFM = model_list.TMFM

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

models_TMFM_narrow = [TMFM_narrow, cmb, sync]
models_TMFM_intermediate = [TMFM_intermediate, cmb, sync]
models_TMFM_broad = [TMFM_broad, cmb, sync]

sMBB_decouple = [models_sMBB, models_sMBB, models_sMBB]
pMBB_broad_decouple = [models_pMBB_broad, models_pMBB_broad, models_pMBB_broad]
TMFM_broad_decouple = [models_TMFM_broad, models_TMFM_broad, models_TMFM_broad]

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

fsigma_T=1e5
fsigma_P=1.

beam_mat = np.identity(3*len(nu_pico)) # Beam model

# pvals set the model parameters
params_sMBB = [sMBB.amp_I, sMBB.amp_Q, sMBB.amp_U, cmb.amp_I, cmb.amp_Q, cmb.amp_U,
              sync.amp_I, sync.amp_Q, sync.amp_U, sMBB.dust_beta, sMBB.dust_T,
              sync.sync_beta]
params_pMBB_narrow = [pMBB_broad.amp_I, pMBB_broad.amp_Q, pMBB_broad.amp_U, cmb.amp_I, cmb.amp_Q, cmb.amp_U,
                sync.amp_I, sync.amp_Q, sync.amp_U, pMBB_broad.dust_beta, pMBB_narrow.dust_T,
                pMBB_broad.sigma_beta, pMBB_broad.sigma_temp, sync.sync_beta]
params_pMBB_intermediate = [pMBB_broad.amp_I, pMBB_broad.amp_Q, pMBB_broad.amp_U, cmb.amp_I, cmb.amp_Q, cmb.amp_U,
                sync.amp_I, sync.amp_Q, sync.amp_U, pMBB_broad.dust_beta, pMBB_narrow.dust_T,
                pMBB_broad.sigma_beta, pMBB_broad.sigma_temp, sync.sync_beta]
params_pMBB_broad = [pMBB_broad.amp_I, pMBB_broad.amp_Q, pMBB_broad.amp_U, cmb.amp_I, cmb.amp_Q, cmb.amp_U,
                sync.amp_I, sync.amp_Q, sync.amp_U, pMBB_broad.dust_beta, pMBB_narrow.dust_T,
                pMBB_broad.sigma_beta, pMBB_broad.sigma_temp, sync.sync_beta]

initial_vals_sMBB = (amp_I, amp_Q, amp_U, cmb.amp_I, cmb.amp_Q, cmb.amp_U,
                    sync.amp_I, sync.amp_Q, sync.amp_U, mean_beta, mean_temp,
                    sync.sync_beta)

initial_vals_sMBB_decouple = (sMBB.amp_I, sMBB.amp_Q, sMBB.amp_U, cmb.amp_I, cmb.amp_Q, cmb.amp_U,
              sync.amp_I, sync.amp_Q, sync.amp_U, sMBB.dust_beta, sMBB.dust_T,
              sync.sync_beta, sMBB.dust_beta, sMBB.dust_T,
              sync.sync_beta, sMBB_U.dust_beta, sMBB_U.dust_T,
              sync.sync_beta)

initial_vals_pMBB_broad = (amp_I, amp_Q, amp_U, cmb.amp_I, cmb.amp_Q, cmb.amp_U,
                        sync.amp_I, sync.amp_Q, sync.amp_U, mean_beta, mean_temp,
                        sigma_beta, sigma_temp, sync.sync_beta)

def batch_fit(models_data, initial_vals, models_fit=models_sMBB, runs=200, nu=nu_pico,
              fsigma_T=fsigma_T, fsigma_P=fsigma_P, decouple=False):
    parent_model = 'mbb'

    logp_max = np.zeros(runs)
    logp_var = np.zeros(runs)
    covs = np.zeros((runs, len(initial_vals), len(initial_vals)))
    best_fits = np.zeros((runs, len(initial_vals)))


    for i in range(runs):
        if i % 10 == 0:
            print('now on run: ', i)

        D_vec, Ninv = fitting.generate_data(nu, fsigma_T, fsigma_P, models_data,
                                            noise_file="data/noise_pico.dat",
                                            decouple=decouple)


        data_spec = (nu_pico, D_vec, Ninv, beam_mat)
        p_spec_sMBB = (pnames_sMBB, initial_vals, parent_model)

        pnames_out, samples, logp  = fitting.joint_mcmc(data_spec,
                                                models_fit, p_spec_sMBB,
                                                nwalkers=48, burn=1000,
                                                steps=10000, nthreads=1,
                                                sample_file=None, decouple=decouple)

        logp_max[i] = logp.max()
        logp_var[i] = logp.var()
        covs[i] = np.cov(samples)
        best_fits[i] = [samples[j][logp.argmax()] for j in range(len(initial_vals))]

    return logp_max, logp_var, best_fits, covs

print('Now fitting sMBB data...')
summary_stats_sMBB = batch_fit(models_sMBB, initial_vals_sMBB)
summary_stats_sMBB_decouple = batch_fit(sMBB_decouple, initial_vals_sMBB_decouple,
                                    decouple=True)

np.savez('summary_stats_sMBB', logp_max_sMBB=summary_stats_sMBB[0],
                               logp_var_sMBB=summary_stats_sMBB[1],
                               best_fits_sMBB=summary_stats_sMBB[2],
                               covs_sMBB=summary_stats_sMBB[3])
np.savez('summary_stats_sMBB_decouple', logp_max_sMBB_decouple=summary_stats_sMBB_decouple[0],
                               logp_var_sMBB_decouple=summary_stats_sMBB_decouple[1],
                               best_fits_sMBB_decouple=summary_stats_sMBB_decouple[2],
                               covs_sMBB_decouple=summary_stats_sMBB_decouple[3])

print('sMBB data fits saved!')

print('Now fitting pMBB data...')
summary_stats_pMBB_narrow = batch_fit(models_pMBB_narrow, initial_vals_sMBB)
summary_stats_pMBB_intermediate = batch_fit(models_pMBB_intermediate, initial_vals_sMBB)
summary_stats_pMBB_broad = batch_fit(models_pMBB_broad, initial_vals_sMBB)
summary_stats_pMBB_decouple = batch_fit(pMBB_broad_decouple, initial_vals_sMBB_decouple,
                                decouple=True)

np.savez('summary_stats_pMBB_narrow', logp_max_pMBB_narrow=summary_stats_pMBB_narrow[0],
                               logp_var_pMBB_narrow=summary_stats_pMBB_narrow[1],
                               best_fits_pMBB_narrow=summary_stats_pMBB_narrow[2],
                               covs_pMBB_narrow=summary_stats_pMBB_narrow[3])

np.savez('summary_stats_pMBB_intermediate', logp_max_pMBB_intermediate=summary_stats_pMBB_intermediate[0],
                               logp_var_pMBB_intermediate=summary_stats_pMBB_intermediate[1],
                               best_fits_pMBB_intermediate=summary_stats_pMBB_intermediate[2],
                               covs_pMBB_intermediate=summary_stats_pMBB_intermediate[3])

np.savez('summary_stats_pMBB_broad', logp_max_pMBB_broad=summary_stats_pMBB_broad[0],
                               logp_var_pMBB_broad=summary_stats_pMBB_broad[1],
                               best_fits_pMBB_broad=summary_stats_pMBB_broad[2],
                               covs_pMBB_broad=summary_stats_pMBB_broad[3])

np.savez('summary_stats_pMBB_decouple', logp_max_pMBB_decouple=summary_stats_pMBB_decouple[0],
                               logp_var_pMBB_decouple=summary_stats_pMBB_decouple[1],
                               best_fits_pMBB_decouple=summary_stats_pMBB_decouple[2],
                               covs_pMBB_decouple=summary_stats_pMBB_decouple[3])

print('pMBB data fits saved!')

print('Now fitting TMFM data...')

summary_stats_TMFM_narrow = batch_fit(models_TMFM_narrow, initial_vals_sMBB)
summary_stats_TMFM_intermediate = batch_fit(models_TMFM_intermediate, initial_vals_sMBB)
summary_stats_TMFM_broad = batch_fit(models_TMFM_broad, initial_vals_sMBB)
summary_stats_TMFM_decouple = batch_fit(TMFM_broad_decouple, initial_vals_sMBB_decouple,
                                       decouple=True)

np.savez('summary_stats_TMFM_narrow', logp_max_TMFM_narrow=summary_stats_TMFM_narrow[0],
                               logp_var_TMFM_narrow=summary_stats_TMFM_narrow[1],
                               best_fits_TMFM_narrow=summary_stats_TMFM_narrow[2],
                               covs_TMFM_narrow=summary_stats_TMFM_narrow[3])

np.savez('summary_stats_TMFM_intermediate', logp_max_TMFM_intermediate=summary_stats_TMFM_intermediate[0],
                               logp_var_TMFM_intermediate=summary_stats_TMFM_intermediate[1],
                               best_fits_TMFM_intermediate=summary_stats_TMFM_intermediate[2],
                               covs_TMFM_intermediate=summary_stats_TMFM_intermediate[3])

np.savez('summary_stats_TMFM_broad', logp_max_TMFM_broad=summary_stats_TMFM_broad[0],
                               logp_var_TMFM_broad=summary_stats_TMFM_broad[1],
                               best_fits_TMFM_broad=summary_stats_TMFM_broad[2],
                               covs_TMFM_broad=summary_stats_TMFM_broad[3])

np.savez('summary_stats_TMFM_decouple', logp_max_TMFM_decouple=summary_stats_TMFM_decouple[0],
                               logp_var_TMFM_decouple=summary_stats_TMFM_decouple[1],
                               best_fits_TMFM_decouple=summary_stats_TMFM_decouple[2],
                               covs_TMFM_decouple=summary_stats_TMFM_decouple[3])

print('TMFM data fits saved!')
