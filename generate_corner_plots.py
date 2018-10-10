import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import model_list, models, fitting
import matplotlib.lines as mlines
import corner
import copy as cp

from utils import rj2cmb

plt.style.use('seaborn-colorblind')
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')

print "hello!"

mean_beta = 1.6
mean_temp = 20.
sigma_beta = .2
sigma_temp = 4.

pMBB_broad = model_list.prob1mbb_model
sMBB = model_list.dust_model
cmb = model_list.cmb_model
sync = model_list.sync_model

DUST_I = 50.
DUST_P = 5. / 1.41
amp_I=rj2cmb(353e9, DUST_I)
amp_Q=rj2cmb(353e9, DUST_P)
amp_U=rj2cmb(353e9, DUST_P)

pMBB_narrow = models.ProbSingleMBB(amp_I=rj2cmb(353e9, DUST_I),
                             amp_Q=rj2cmb(353e9, DUST_P),
                             amp_U=rj2cmb(353e9, DUST_P),
                             dust_beta=1.6, dust_T=20.,
                             sigma_beta=.1 * sigma_beta, sigma_temp=.1 * sigma_temp)

nu_pico = np.asarray([21,25,30, 36.0,43.2,51.8,62.2,74.6,89.6,
                      107.5,129.0,154.8,185.8,222.9,267.5,321.0,
                      385.2,462.2,554.7,665.6,798.7]) * 1e9

models_sMBB = [sMBB, cmb, sync]
models_pMBB_broad = [pMBB_broad, cmb, sync]
models_pMBB_narrow = [pMBB_narrow, cmb, sync]

def make_pnames(models_fit):
    amp_names = []
    param_names = []

    for mod in models_fit:
        # Parameter names
        amp_names += ["%s_%s" % (mod.model, pol) for pol in "IQU"]
        param_names += mod.param_names

    return amp_names + param_names

pnames_sMBB = make_pnames(models_sMBB)
pnames_pMBB_broad = make_pnames(models_pMBB_broad)
pnames_pMBB_narrow = make_pnames(models_pMBB_narrow)
print pnames_sMBB

fsigma_T=1e3
fsigma_P=1.

beam_mat = np.identity(3*len(nu_pico)) # Beam model

# pvals set the model parameters
params_sMBB = [sMBB.amp_I, sMBB.amp_Q, sMBB.amp_U, cmb.amp_I, cmb.amp_Q, cmb.amp_U,
              sync.amp_I, sync.amp_Q, sync.amp_U, sMBB.dust_beta, sMBB.dust_T,
              sync.sync_beta]
params_pMBB_broad = [pMBB_broad.amp_I, pMBB_broad.amp_Q, pMBB_broad.amp_U, cmb.amp_I, cmb.amp_Q, cmb.amp_U,
                  sync.amp_I, sync.amp_Q, sync.amp_U, pMBB_broad.dust_beta, pMBB_broad.dust_T,
                  pMBB_broad.sigma_beta, pMBB_broad.sigma_temp, sync.sync_beta]
params_pMBB_narrow = [pMBB_narrow.amp_I, pMBB_narrow.amp_Q, pMBB_narrow.amp_U, cmb.amp_I, cmb.amp_Q, cmb.amp_U,
                  sync.amp_I, sync.amp_Q, sync.amp_U, pMBB_narrow.dust_beta, pMBB_narrow.dust_T,
                  pMBB_narrow.sigma_beta, pMBB_narrow.sigma_temp, sync.sync_beta]

initial_vals_sMBB = (amp_I, amp_Q, amp_U, cmb.amp_I, cmb.amp_Q, cmb.amp_U,
                    sync.amp_I, sync.amp_Q, sync.amp_U, mean_beta, mean_temp,
                    sync.sync_beta)
initial_vals_pMBB_broad = (amp_I, amp_Q, amp_U, cmb.amp_I, cmb.amp_Q, cmb.amp_U,
                        sync.amp_I, sync.amp_Q, sync.amp_U, mean_beta, mean_temp,
                        sigma_beta, sigma_temp, sync.sync_beta)
initial_vals_pMBB_narrow = (amp_I, amp_Q, amp_U, cmb.amp_I, cmb.amp_U, cmb.amp_Q,
                        sync.amp_I, sync.amp_Q, sync.amp_U,mean_beta, mean_temp,
                        .1 * sigma_beta, .1 * sigma_temp, sync.sync_beta)
parent_model = 'mbb'

D_vec_sMBB, Ninv = fitting.generate_data(nu_pico, fsigma_T, fsigma_P, [sMBB, cmb, sync],
                                        noise_file="data/noise_pico.dat" )
D_vec_pMBB_broad, Ninv = fitting.generate_data(nu_pico, fsigma_T, fsigma_P, [pMBB_broad, cmb, sync],
                                           noise_file="data/noise_pico.dat")
D_vec_pMBB_narrow, Ninv = fitting.generate_data(nu_pico, fsigma_T, fsigma_P, [pMBB_narrow, cmb, sync],
                                           noise_file="data/noise_pico.dat")

data_spec_sMBB = (nu_pico, D_vec_sMBB, Ninv, beam_mat)
data_spec_pMBB_broad = (nu_pico, D_vec_pMBB_broad, Ninv, beam_mat)
data_spec_pMBB_narrow = (nu_pico, D_vec_pMBB_narrow, Ninv, beam_mat)

p_spec_sMBB = (pnames_sMBB, initial_vals_sMBB, parent_model)
p_spec_pMBB_broad = (pnames_pMBB_broad, initial_vals_pMBB_broad, parent_model)
p_spec_pMBB_narrow = (pnames_pMBB_narrow, initial_vals_pMBB_narrow, parent_model)

print "running emcee"

pnames_out_sMBB, samples_sMBB, logp_sMBB  = fitting.joint_mcmc(data_spec_sMBB, [sMBB, cmb, sync], p_spec_sMBB, nwalkers=30,
               burn=1000, steps=10000, nthreads=8, sample_file=None)

#pnames_out_pMBB_broad, samples_pMBB_broad, logp_pMBB_broad  = fitting.joint_mcmc(data_spec_pMBB_broad, [sMBB, cmb, sync], p_spec_sMBB, nwalkers=30,
#               burn=1000, steps=10000, nthreads=8, sample_file=None)

#pnames_out_pMBB_narrow, samples_pMBB_narrow, logp_pMBB_narrow  = fitting.joint_mcmc(data_spec_pMBB_narrow, [sMBB, cmb, sync], p_spec_sMBB, nwalkers=30,
#               burn=1000, steps=10000, nthreads=8, sample_file=None)

ax1 = corner.corner(samples_sMBB.T, labels=['dust I', 'dust Q', 'dust U', 'cmb I', 'cmb Q', 'cmb U', 'sync I', 'sync Q', 'sync U', 'dust beta', 'dust temp', 'sync beta'], truths=initial_vals_sMBB, plot_datapoints=False)

#ax2 = corner.corner(samples_pMBB_broad.T, labels=['dust I', 'dust Q', 'dust U', 'cmb I', 'cmb Q', 'cmb U', 'sync I', 'sync Q', 'sync U',
 #                          'mean beta', 'mean temp', 'sync beta'],
 #                          truths=initial_vals_sMBB, plot_datapoints=False)

#ax3 = corner.corner(samples_pMBB_narrow.T, labels=['dust I', 'dust Q', 'dust U', 'cmb I', 'cmb Q', 'cmb U', 'sync I', 'sync Q', 'sync U',
#                           'mean beta', 'mean temp', 'sync beta'],
#                           truths=initial_vals_sMBB, plot_datapoints=False)

ax1.savefig('sMBB2sMBBtestpdf')
#ax2.savefig('sMBB2pMBB_broad.pdf')
#ax3.savefig('sMBB2pMBB_narrow.pdf')
