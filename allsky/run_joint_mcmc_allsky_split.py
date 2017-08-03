#!/usr/bin/python
"""
Do MCMC runs to fit FG models to simulated data
"""
import numpy as np
import models
import model_list_allsky
import fitting
from utils import rj2cmb, bands_log
import sys, time, os, copy
import healpy as hp

import pdb

def main( in_list = [ 'cmb', 'sync', 'mbb' ],
          fit_list = [ 'cmb', 'sync', 'mbb' ],
          nu_min = 15,
          nu_max = 800,
          n_bands = 7,
          input_dict = [],
          idx_px = 0,
          seed = 100,
          Nside = 16 ):

    # 1) Basic setup for the run
    # Prefix for output files
    PREFIX = "final"
    NBURN = 500 # 500, 50 (test)
    NSTEPS = 10000 # 10000, 25 (test)
    NWALKERS = 100
    # Used to produce summary stats from some burn-in cuts
    cut_range = np.arange( NSTEPS, step = int( np.round( NSTEPS * 0.8 ) ) )
    # Reference noise curve (assumes noise file contains sigma_P=sigma_Q=sigma_U 
    # in uK_CMB.deg for a given experiment)
    NOISE_FILE = "../data/noise_coreplus_extended.dat"
    # Number of threads used in the MCMC
    nthreads = 1 # Kept as 1 here because of the MPI use in the launch_allsky
    # Set random seed
    np.random.seed( seed )

    # Define input models and their amplitudes/parameters
    allowed_comps = model_list_allsky.model_dict( fg_dict = input_dict, idx_px = idx_px )
    cmb_model = model_list_allsky.cmb_model()

    # Frequency range. NB: in the all-sky version, the frequency range is unique, and different frequency ranges are run independently
    numin_vals = [ nu_min ]
    numax_vals = [ nu_max ]

    # Temperature/polarisation noise rms for all bands, as a fraction of T_cmb
    fsigma_T = 1. / np.sqrt(2.)
    fsigma_P = 1.

    # Collect components into lists and set input amplitudes
    mods_in = [ allowed_comps[ comp ] for comp in in_list ]
    mods_fit = [ allowed_comps[ comp ] for comp in fit_list ]
    amps_in = np.array( [ m.amps() for m in mods_in ] )
    params_in = np.array( [ m.params() for m in mods_in ] )
    # This is helpful when creating the map where the results will be stored
    amps_fit = np.array( [ m.amps() for m in mods_fit ] )
    params_fit = np.array( [ m.params() for m in mods_fit ] )

    # Expand into all combinations of nu_min,max. NB: this is how the single pixel case was working. Kept if necessary in the future.
    nu_min, nu_max = np.meshgrid(numin_vals, numax_vals)
    nu_params = np.column_stack((nu_min.flatten(), nu_max.flatten()))

    def model_test(nu, D_vec, Ninv, models_fit, initial_vals=None, burn=500,
                   steps=1000, nwalkers=100, cmb_amp_in=None, sample_file=None, 
		   i_px = 0, nthreads = 1 ):
        """
        Generate simulated data given an input model, and perform MCMC fit using
        another model.
        """
        # Collect together data and noise/instrument model
        beam_mat = np.identity(3*len(nu)) # Beam model
        data_spec = (nu, D_vec, Ninv, beam_mat)

        # Get a list of amplitude/parameter names and initial values
        amp_names = []; amp_vals = []; param_names = []; param_vals = []
        amp_parent_model = []; param_parent_model = []
        for mod in models_fit:
                # Parameter names
                amp_names += ["%s_%s" % (mod.model, pol) for pol in "IQU"]
                param_names += mod.param_names

                # Parameter values
                amp_vals = np.concatenate( (amp_vals, mod.amps()) )
                param_vals = np.concatenate( (param_vals, mod.params()) )

                # Parent model list
                amp_parent_model.append(mod)
                param_parent_model.append(mod)

        # Concatenate parameter lists
        pnames = amp_names + param_names
        pvals = np.concatenate((amp_vals, param_vals))
        parent_model = amp_parent_model + param_parent_model

        # Use 'guess' as the initial point for the MCMC if specified
        if initial_vals is None: initial_vals = pvals

        # Collect names, initial values, and parent components for the parameters
        param_spec = (pnames, initial_vals, parent_model)

        # Run MCMC sampler on this model
        t0 = time.time()
        pnames, samples, logp = fitting.joint_mcmc(data_spec, models_fit, param_spec,
                                                   burn=burn, steps=steps,
                                                   nwalkers=nwalkers,
                                                   nthreads=nthreads,
                                                   sample_file=sample_file)
        if i_px == 0:
            print "MCMC run in %d sec." % (time.time() - t0)

        # Return parameter names and samples
        return pnames, samples, logp, initial_vals

    def run_model(nu_params, Nside, i_px ):

        ## Number of maps
        n_maps = 0
        for ii in range( len( amps_fit ) ):
                n_maps += len( amps_fit[ ii ] )
        for jj in range( len( params_fit ) ):
                n_maps += len( params_fit[ jj ] )

        # Get band definition
        nu_min, nu_max = nu_params
        #print "nu_min = %d GHz, nu_max = %d GHz" % (nu_min, nu_max)
        nu = bands_log(nu_min, nu_max, n_bands)
        label = str(nu_min) + '_' + str(nu_max)

        # Make copies of models
        my_mods_in = mods_in
        my_mods_fit = mods_fit

        fname_samples = None

        # Simulate data and run MCMC fit
        D_vec, Ninv = fitting.generate_data(nu, fsigma_T, fsigma_P,
                                                components=my_mods_in,
                                                noise_file=NOISE_FILE, idx_px = i_px)

        pnames, samples, logp, ini = model_test(nu, D_vec, Ninv, my_mods_fit,
                                                burn=NBURN, steps=NSTEPS,
                                                nwalkers=NWALKERS,
                                                cmb_amp_in=cmb_model.amps(),
                                                sample_file=fname_samples, 
						i_px = i_px, 
						nthreads = nthreads )
        # Calculate best-fit chisq.
        chisq = -2.*logp
        dof = D_vec.size - len(pnames)

        # Reshape sample array into (Nparams, Nwalkers, Nsamples)
        samples = samples.reshape((samples.shape[0],
                                   NWALKERS,
                                   samples.shape[1]/NWALKERS))

        # Array to store the results (NB: this is the structure for the final FITS file to be stored with healpy)
        data_fits = np.full( ( len( pnames ) * 3 ), hp.UNSEEN, dtype = 'float32' )

        # Loop over different burn-in cuts to produce summary stats
        ## Right now, since it's only printing out some diagnostic, it's set for one case only
        ## Before: for cut in cut_range:
        for cut in [ cut_range[ -1 ] ]:

                # Output mean and bias
                summary_str = "%4.4e %4.4e %4.4e " % (nu_min, nu_max, np.min(chisq))
                summary_data = [nu_min, nu_max, np.min(chisq)]
                header = "nu_min nu_max chi2_min "
                # Loop over parameter names
                for i in range(len(pnames)):

                        # Mean, std. dev., and fractional shift from true value
                        _mean = np.mean(samples[i,:,cut:])
                        _std = np.std(samples[i,:,cut:])
                        _fracbias = (np.mean(samples[i,:,cut:]) - ini[i]) \
                                          / np.std(samples[i,:,cut:])
                        stats = [_mean, _std, _fracbias]
                        data_fits[ 3 * i ] = ini[ i ] 
                        data_fits[ 3 * i + 1 ] = _mean 
                        data_fits[ 3 * i + 2 ] = _std

                        # Keep summary stats, to be written to file
                        summary_data += stats
                        header += "mean_%s std_%s Delta_%s " \
                                        % (pnames[i], pnames[i], pnames[i])
                        # Printing out only the first pixel
                        if i_px == 0:
                            print "%14s: %+3.3e %+3.3e +/- %3.3e [Delta = %+3.3f]" \
                            % (pnames[i], ini[ i ], stats[ 0 ], stats[ 1 ], stats[ 2 ] )
        return data_fits

    # Run the model for this set of params
    return run_model( nu_params[ 0 ], Nside, idx_px )

if __name__ == '__main__':
     main()
