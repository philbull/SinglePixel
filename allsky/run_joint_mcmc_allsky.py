#!/usr/bin/python
"""
Do MCMC runs to fit FG models to simulated data, over a grid of 
(nu_min, nu_max) values.
"""
import numpy as np
import models
import model_values_allsky
import model_list_allsky
import fitting
from utils import rj2cmb, bands_log
import sys, time, os, copy
import healpy as hp

import pdb

#from multiprocessing import Pool
##from mpi4py import MPI

# Set-up MPI
##comm = MPI.COMM_WORLD
##myid = comm.Get_rank()
##nproc = comm.Get_size()
## (Sergi) PS: If MPI gets used at the top level of sending independent single pixel jobs, one should deal with the map results I/O in process 0

# Prefix for output files
PREFIX = "final"
NBURN = 50 #500
NSTEPS = 25 #10000
NWALKERS = 100

# Reference noise curve (assumes noise file contains sigma_P=sigma_Q=sigma_U 
# in uK_CMB.deg for a given experiment)
NOISE_FILE = "../data/noise_coreplus_extended.dat"
#NOISE_FILE = "data/core_plus_extended_noise.dat"

# Band parameter definitions
nbands = 7
nthreads = 2

# Set random seed
SEED = 10
if len(sys.argv) > 1:
    SEED = int(sys.argv[1])
print "SEED =", SEED
np.random.seed(SEED)

# Lists of input and output models
in_list = ['cmb', ]; fit_list = ['cmb', ]
# Defaut list of allowed models
allowed_comps = model_list_allsky.model_dict() 
# Parse args to define input and output models
if len(sys.argv) > 2:
    # Removing any white spaces and creating a list
    in_list_argv = "".join( sys.argv[ 2 ].split( ) )
    in_list += in_list_argv.split(",")
    fit_list_argv = "".join( sys.argv[ 3 ].split( ) )
    fit_list += fit_list_argv.split(",")
else:
    in_list = ['cmb', 'sync', 'mbb']
    fit_list = ['cmb', 'sync', 'mbb']

# Make sure models are of known types
for item in in_list:
    if item not in allowed_comps.keys():
        raise ValueError("Unknown component type '%s'" % item)

for item in fit_list:
    if item not in allowed_comps.keys():
        raise ValueError("Unknown component type '%s'" % item)

# Print recognised models and specify name
print "Input components:", in_list
print "Fitting components:", fit_list
name_in = "-".join(in_list)
name_fit = "-".join(fit_list)

# Getting the Nside and pixel number (ring scheme will be followed, as it is the default ordering scheme in Python)
if len( sys.argv ) > 4:
    if len( sys.argv ) != 6: 
        sys.exit( '(run_joint_mcmc_allsky) Provide Nside *and* a pixel number. Exiting.' )
    Nside = int( sys.argv[ 4 ] )
    Npix = int( sys.argv[ 5 ] )
    if ( Npix >= 12 * Nside * Nside ):
        sys.exit( '(run_joint_mcmc_allsky) Pixel number exceeds 12*Nside*Nside. Exiting' )
else:
    Nside = 8 
    Npix = 0

# Define input models and their amplitudes/parameters
allowed_comps = model_list_allsky.model_dict( fg_dict = model_values_allsky.get_dict( in_list, Nside, Npix ) ) 
cmb_model = model_list_allsky.cmb_model()

# Frequency ranges
numin_vals = [ 15. ] #[15., 20., 25., 30., 35., 40.]
numax_vals = [ 700. ] #[300., 400., 500., 600., 700., 800.]

# Temperature/polarisation noise rms for all bands, as a fraction of T_cmb
fsigma_T = 1. / np.sqrt(2.)
fsigma_P = 1.

# Collect components into lists and set input amplitudes
mods_in = [allowed_comps[comp] for comp in in_list]
mods_fit = [allowed_comps[comp] for comp in fit_list]
amps_in = np.array([m.amps() for m in mods_in])
params_in = np.array([m.params() for m in mods_in])

# Expand into all combinations of nu_min,max
nu_min, nu_max = np.meshgrid(numin_vals, numax_vals)
nu_params = np.column_stack((nu_min.flatten(), nu_max.flatten()))

# Prepare output files for writing
# Create the directory if it does not exist
out_dir = 'output'
if not os.path.exists( out_dir ):
    os.makedirs( out_dir ) # PS: one could use exist_ok in python v3.7

filename = "%s/%s_summary_%s.%s_nb%d_seed%d_nside%04i" \
             % ( out_dir, PREFIX, name_in, name_fit, nbands, SEED, Nside )

cut_range = np.arange(NSTEPS, step=200)
for cut in cut_range:
    fname = filename + "_cut%d.dat" % cut
    # Commented out for now. If uncommented, search for 'summary statistics'
    ##f = open(fname, 'w')
    ##f.close()

def model_test(nu, D_vec, Ninv, models_fit, initial_vals=None, burn=500, 
               steps=1000, nwalkers=100, cmb_amp_in=None, sample_file=None):
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
    print "MCMC run in %d sec." % (time.time() - t0)
    
    # Return parameter names and samples
    return pnames, samples, logp, initial_vals


def run_model(nu_params, Nside, Npix):

    ## Number of maps
    n_maps = ( amps_in.size + params_in.size ) * 3   
    if os.path.isfile( filename + '.fits' ) == 1:
        # Check whether there are already some results (if the fit was forced to be re-started, the values are hp.UNSEEN)  
        data_ini = hp.read_map( filename + '.fits', range( n_maps ), verbose = False )
        # if any is hp.UNSEEN, run the fit, otherwise return
        s_tmp = 0 ;
        for i in range( n_maps ): s_tmp += np.abs( data_ini[ i ][ Npix ] )
        if s_tmp < np.abs( hp.UNSEEN ): 
            print '(run_joint_mcmc_allsky) Pixel %i already analyzed. Skipping it.' % ( Npix )
            return
    # creare the output file if it does not exist
    if os.path.isfile( filename + '.fits' ) != 1:
        data_start = np.full( ( n_maps, 12 * Nside * Nside ), hp.UNSEEN )
        hp.write_map( filename + '.fits', data_start, dtype = 'float32' )    

    # Get band definition
    nu_min, nu_max = nu_params
    print "nu_min = %d GHz, nu_max = %d GHz" % (nu_min, nu_max)
    nu = bands_log(nu_min, nu_max, nbands)
    label = str(nu_min) + '_' + str(nu_max)
    
    # Make copies of models
    my_mods_in = mods_in
    my_mods_fit = mods_fit
    
    # Name of sample file
    #fname_samples = "output/%s_samples_%s.%s_nb%d_seed%d_%s.dat" \
    #              % (PREFIX, name_in, name_fit, nbands, SEED, label)
    fname_samples = None
    
    # Simulate data and run MCMC fit
    D_vec, Ninv = fitting.generate_data(nu, fsigma_T, fsigma_P, 
                                        components=my_mods_in, 
                                        noise_file=NOISE_FILE)
                                        
    pnames, samples, logp, ini = model_test(nu, D_vec, Ninv, my_mods_fit, 
                                            burn=NBURN, steps=NSTEPS, 
                                            nwalkers=NWALKERS,
                                            cmb_amp_in=cmb_model.amps(),
                                            sample_file=fname_samples)
    # Calculate best-fit chisq.
    chisq = -2.*logp
    dof = D_vec.size - len(pnames)
    
    # Reshape sample array into (Nparams, Nwalkers, Nsamples)
    samples = samples.reshape((samples.shape[0], 
                               NWALKERS, 
                               samples.shape[1]/NWALKERS))
    
    
    # Loop over different burn-in cuts to produce summary stats
    for cut in cut_range:
        
        # Set output filename
        fname = filename + "_cut%d.dat" % cut
        
        # Output mean and bias
        summary_str = "%4.4e %4.4e %4.4e " % (nu_min, nu_max, np.min(chisq))
        summary_data = [nu_min, nu_max, np.min(chisq)]
        header = "nu_min nu_max chi2_min "
        data_fits = np.full( ( len( pnames ), 3 ), hp.UNSEEN, dtype = 'float32' )
        # Loop over parameter names
        for i in range(len(pnames)):
            
            # Mean, std. dev., and fractional shift from true value
            _mean = np.mean(samples[i,:,cut:])
            _std = np.std(samples[i,:,cut:])
            _fracbias = (np.mean(samples[i,:,cut:]) - ini[i]) \
                      / np.std(samples[i,:,cut:])
            stats = [_mean, _std, _fracbias]
            data_fits[ i ] = [ ini[ i ], _mean, _std ]
            
            # Keep summary stats, to be written to file
            summary_data += stats
            header += "mean_%s std_%s Delta_%s " \
                    % (pnames[i], pnames[i], pnames[i])
            
            print "%14s: %+3.3e %+3.3e +/- %3.3e [Delta = %+3.3f]" \
                  % (pnames[i], ini[ i ], stats[ 0 ], stats[ 1 ], stats[ 2 ] )

        # Insert the results (each pixel writes a unique location, but remember the comment at the top if MPI gets used for all pixels) 
        data_maps = hp.read_map( filename + '.fits', range( len( pnames ) * 3 ), verbose = False )
        for i_tmp in range( len( pnames) ): 
           for i_tmp_2 in range( 3 ): data_maps[ i_tmp * 3 + i_tmp_2 ][ Npix ] = data_fits[ i_tmp ][ i_tmp_2 ]
        hp.write_map( filename + '.fits', data_maps, dtype = 'float32' )
        
        ## Commented out for now. PS: if uncommented, create the file as it was done search for 'f = open(fname, 'w')'
        # If file is empty, set flag to write header when saving output
        ##has_header = False if os.stat(fname).st_size == 0 else True
        
        # Append summary statistics to file
        ##f = open(fname, 'a')
        ##if has_header:
        ##    np.savetxt(f, np.atleast_2d(summary_data))
        ##else:
        ##    np.savetxt(f, np.atleast_2d(summary_data), header=header[:-1])
        ##f.close()
        
################################################
# Running the loop over frequency combinations #
################################################

for i in range(len(nu_params)):
    # Run the model for this set of params
    run_model(nu_params[i], Nside, Npix)

