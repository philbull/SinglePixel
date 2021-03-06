
import numpy as np
from scipy.linalg import sqrtm
from scipy.interpolate import interp1d
import copy, time
import emcee

def ln_prior(pvals, models):
    """
    Logarithm of the prior (mostly just uniform prior bounds for now).
    """
    # Define priors
    priors = {
        'prob1mbb_Q':   (1., 100.),
        'prob1mbb_U':   (1., 100.),
        'dust_T':       (16., 24.),
        'dust_beta':    (1.4, 1.8),
        'sync_beta':    (-1.6, -0.8),
        'ame_nupeak':   (15., 35.),
        'gdust_beta':   (1.1, 1.8),
        'gdust_dbeta':  (-1.8, 1.8),
        'gdust_Td1':    (5., 30.),
        'gdust_Td2':    (5., 30.),
        'gdust_fI':     (0., 1.),
        'gdust_fQ':     (-2., 2.),
        'gdust_fU':     (-2., 2.),
        'sigma_beta':   (1e-2, 1.),
        'sigma_temp':   (.1, 10.),
    }

    # Make ordered list of parameter names
    param_names = []
    for mod in models:
        param_names += mod.param_names

    # Get a list of amplitude names
    amp_names = []
    for mod in models:
        amp_names += ["%s_%s" % (mod.model, pol) for pol in "IQU"]

    pnames = amp_names + param_names

    # Go through priors and apply them
    ln_prior = 0. # Set default prior value
    for pn in priors.keys():
    #    print 'pnames: ' + str(pnames)
    #    print 'pn: ' + str(pn)
        if pn not in pnames: continue
        pmin, pmax = priors[pn] # Prior bounds
    #    print 'pmin: ' + str(pmin) + ', pmax: ' + str(pmax)
    #    print 'pvals: ' + str(pvals)
        val = pvals[pnames.index(pn)] # Current value of parameter
    #    print 'val: ' + str(val)
        if val < pmin or val > pmax:
            ln_prior = -np.inf
    return ln_prior

def lnprob(pvals, data_spec, models_fit, param_spec, Ninv_sqrt):
    """
    log-probability (likelihood times prior) for a set of parameter values.
    """
    # Retrieve instrument/data model and parameter info
    nu, D_vec, Ninv, beam_mat = data_spec

    # pnames should be amps_names + param_names
    # initial_vals should be amps_vals + param_vals
    pnames, initial_vals, parent_model = param_spec

    # Apply prior
    logpr = ln_prior(pvals, models_fit)
    if not np.isfinite(logpr):
        return -np.inf

    F_fg, F_cmb, F = F_matrix(pvals, nu, models_fit, param_spec)
    H = F_fg.T * Ninv * F_fg

    # GLS solution for component amplitudes
    x_mat = np.linalg.inv(F.T * beam_mat.T * Ninv * beam_mat * F) \
          * F.T * beam_mat.T * Ninv * D_vec # Equation A3

    chi_square = (D_vec - beam_mat * F * x_mat).T * Ninv \
               * (D_vec - beam_mat * F * x_mat) # Equation A4

    # Equation A14
    U, Lambda, VT = np.linalg.svd(Ninv_sqrt*F_fg, full_matrices=False)

    # Equation A16
    N_eff_inv_cmb = F_cmb.T * Ninv_sqrt \
                  * (np.matrix(np.identity(U.shape[0])) - U*U.T) \
                  * Ninv_sqrt * F_cmb

    # Total log posterior
    lnprob = logpr - chi_square - 0.5*np.log(np.linalg.det(H)) \
                              - 0.5*np.log(np.linalg.det(N_eff_inv_cmb))

    # Return log-posterior and GLS amplitudes
    return lnprob, np.array(x_mat.T)[0]


def lnprob_joint(params, data_spec, models_fit, param_spec):
    """
    log-probability (likelihood times prior) for a set of parameter values.
    """
    # Retrieve instrument/data model and parameter info
    nu, D_vec, Ninv, beam_mat = data_spec
    pnames, initial_vals, parent_model = param_spec
    Nmod = len(models_fit)
    Npol = 3

    # Separate amplitude and spectral model parameters
    amps = params[:Nmod*Npol]
    pvals = params[Nmod*Npol:]

    # Apply prior
    logpr = ln_prior(params, models_fit)
    if not np.isfinite(logpr):
        return -np.inf

    # Create new copies of model objects to work with
    #models = [copy.deepcopy(m) for m in models_fit]
    models = models_fit

    # Set new parameter values for the copied model objects, and then get
    # scalings as a function of freq./polarisation
    pstart = 0
    mdata = np.zeros(nu.size * Npol)
    for i in range(len(models)):
        m = models[i]

        # Set new parameter values in the models
        n = m.params().size
        #m.set_params( pvals[pstart:pstart+n] )
        mparams = pvals[pstart:pstart+n]
        pstart += n # Increment for next model

        # Calculate scaling with freq. given new parameter values
        amp = np.outer( amps[3*i:3*(i+1)], np.ones(nu.size) ) # Npol*Nfreq array

        # Apply positivity prior on I amplitudes of all components
        #if m.model == 'ame':
        if np.any(amp[0] < 0.):
            return -np.inf

        # Add to model prediction of data vector
        mdata += (amp * m.scaling(nu, params=mparams)).flatten()

    # Calculate chi-squared with data (assumed beam = 1)
    mdata = np.matrix(mdata).T
    chi_square = (D_vec - mdata).T * Ninv * (D_vec - mdata)

    # Return log-posterior
    #return logpr - 0.5 * chi_square
    return -0.5 * chi_square


def F_matrix(pvals, nu, models_fit, param_spec):
    """
    Foreground spectral dependence operator.
    """
    pnames, initial_vals, parent_model = param_spec

    # Check that the CMB component is the first component in the model list
    if models_fit[0].model != 'cmb':
        raise ValueError("The first model in the models_fit list should be a "
                         "CMB() object.")

    Nband = len(nu) # No. of frequency bands
    Npol = 3 # No. of data components (I, Q, U)
    Ncomp = len(models_fit) # No. of sky components

    F_fg = np.zeros((Npol * Nband, Npol * (Ncomp - 1)))
    F_cmb = np.zeros((Npol * Nband, Npol))
    F = np.zeros((Npol * Nband, Npol * Ncomp))

    # Create new copies of model objects to work with
    #models = [copy.deepcopy(m) for m in models_fit]
    models = models_fit

    # Set new parameter values for the copied model objects, and then get
    # scalings as a function of freq./polarisation
    pstart = 0; k = -1
    for i in range(len(models)):
        m = models[i]

        # Set new parameter values in the models
        n = m.params().size
        #m.set_params( pvals[pstart:pstart+n] )
        mparams = pvals[pstart:pstart+n]
        pstart += n # Increment for next model
        if m.model != 'cmb': k += 1 # Increment for next non-CMB model

        # Calculate scaling with freq. given new parameter values
        scal = m.scaling(nu, params=mparams)

        for j in range(Npol):
            # Fill FG or CMB -matrix with scalings, as appropriate
            if m.model != 'cmb':
                F_fg[j*Nband:(j+1)*Nband, k*Npol + j] = scal[j,:]
            else:
                F_cmb[j*Nband:(j+1)*Nband, j] = scal[j,:]

    # Stack CMB and FG F-matrices together
    F = np.hstack((F_cmb, F_fg))
    return np.matrix(F_fg), np.matrix(F_cmb), np.matrix(F)


def mcmc(data_spec, models_fit, param_spec, nwalkers=50,
         burn=500, steps=1000, sample_file=None):
    """
    Run MCMC to fit model to some simulated data.
    """
    # Retrieve instrument/data model and parameter info
    nu, D_vec, Ninv, beam_mat = data_spec
    pnames, initial_vals, parent_model = param_spec

    # Invert noise covariance matrix
    Ninv_sqrt = np.matrix(sqrtm(Ninv))

    # Get a list of model parameter names (FIXME: Ignores input pnames for now)
    param_names = []
    for mod in models_fit:
        param_names += mod.param_names

    # Get a list of amplitude names
    fg_amp_names = []; cmb_amp_names = []
    for mod in models_fit:
        if mod.model == 'cmb':
            cmb_amp_names += ["%s_%s" % (mod.model, pol) for pol in "IQU"]
        else:
            fg_amp_names += ["%s_%s" % (mod.model, pol) for pol in "IQU"]
    pnames = cmb_amp_names + fg_amp_names + param_names

    # Define starting points
    ndim = len(initial_vals)
    pos = [initial_vals*(1.+1e-3*np.random.randn(ndim)) for i in range(nwalkers)]

    # Run emcee sampler
    sampler = emcee.EnsembleSampler( nwalkers, ndim, lnprob,
                           args=(data_spec, models_fit, param_spec, Ninv_sqrt) )
    sampler.run_mcmc(pos, burn + steps)

    # Recover samples of spectral parameters and amplitudes
    param_samples = sampler.chain[:, burn:, :].reshape((-1, ndim))
    amp_samples = np.swapaxes(np.array(sampler.blobs), 0, 1)
    amp_samples = amp_samples[:, burn:, :].reshape((-1, amp_samples.shape[2]))
    samples = np.concatenate((amp_samples.T, param_samples.T))

    # Save chains to file
    if sample_file is not None:
        np.savetxt(sample_file, samples, fmt="%.6e", header=" ".join(pnames))

    # Summary statistics for fitted parameters
    params_out = np.median(param_samples, axis=0)

    # Return summary statistics and samples
    return params_out, pnames, samples


def joint_mcmc(data_spec, models_fit, param_spec, nwalkers=100,
               burn=500, steps=1000, nthreads=2, sample_file=None):
    """
    Run MCMC to fit model to some simulated data. Fits to all parameters, both
    amplitudes and spectral parameters.
    """
    # Retrieve instrument/data model and parameter info
    nu, D_vec, Ninv, beam_mat = data_spec
    pnames, initial_vals, parent_model = param_spec

    # Get a list of model parameter names (FIXME: Ignores input pnames for now)
    param_names = []
    for mod in models_fit:
        param_names += mod.param_names

    # Get a list of amplitude names
    amp_names = []
    for mod in models_fit:
        amp_names += ["%s_%s" % (mod.model, pol) for pol in "IQU"]
    pnames = amp_names + param_names

    # Define starting points
    ndim = len(initial_vals)
    pos = [initial_vals*(1.+1e-3*np.random.randn(ndim)) for i in range(nwalkers)]

    #print param_spec
    # Run emcee sampler
    sampler = emcee.EnsembleSampler( nwalkers, ndim, lnprob_joint,
                                     args=(data_spec, models_fit, param_spec),
                                     threads=nthreads )
    sampler.run_mcmc(pos, burn + steps)

    # Recover samples of spectral parameters and amplitudes
    samples = sampler.chain[:, burn:, :].reshape((-1, ndim))

    # Recover log(posterior)
    logp = sampler.lnprobability[:,burn:].reshape((-1,))

    # Save chains to file
    if sample_file is not None:
        np.savetxt(sample_file, samples, fmt="%.6e", header=" ".join(pnames))

    # Return summary statistics and samples
    return pnames, samples.T, logp


def noise_model_old(fname="data/CMBpol_extended_noise.dat", scale=1.):
    """
    Load noise model from file and create interpolation function as a fn of
    frequency. This is the noise per pixel, for some arbitrary pixel size.
    """
    # Load from file
    nu, sigma = np.genfromtxt(fname).T

    # Extrapolate at the ends of the frequency range
    if nu[0] > 1.:
        sigma0 = sigma[0] \
               + (sigma[1] - sigma[0]) / (nu[1] - nu[0]) * (1. - nu[0])
        sigman = sigma[-1] \
               + (sigma[-1] - sigma[-2]) / (nu[-1] - nu[-2]) * (1e3 - nu[-1])
        if sigma0 < 0.: sigma0 = sigma[0]
        if sigman < 0.: sigman = sigma[-1]

        # Add to end of range
        nu = np.concatenate(([1.,], nu, [1e3,]))
        sigma = np.concatenate(([sigma0,], sigma, [sigman,]))

    # Rescale by constant overall factor
    sigma *= scale

    # Construct interpolation function
    return interp1d(nu, sigma, kind='linear', bounds_error=False)


def noise_model(fname="data/noise_coreplus_extended.dat", scale=1.):
    """
    Load noise model from file and create interpolation function as a fn of
    frequency. This is the noise per pixel, for some arbitrary pixel size.
    """
    # Load from file
    dat = np.genfromtxt(fname).T
    if dat.shape[0] == 3:
        nu, fwhm, sigma = dat
    elif dat.shape[0] == 2:
        nu, sigma = dat
    else:
        raise ValueError("Unexpected number of columns in noise file.")

    # Rescale by constant overall factor
    sigma *= scale

    # Work in log-space
    sigma = np.log(sigma)

    # Extrapolate at the ends of the frequency range
    if nu[0] > 1.:
        sigma0 = sigma[0] \
               + (sigma[1] - sigma[0]) / (nu[1] - nu[0]) * (1. - nu[0])
        sigman = sigma[-1] \
               + (sigma[-1] - sigma[-2]) / (nu[-1] - nu[-2]) * (1e3 - nu[-1])
        if sigma0 < 0.: sigma0 = sigma[0]
        if sigman < 0.: sigman = sigma[-1]

        # Add to end of range
        nu = np.concatenate(([1.,], nu, [1e3,]))
        sigma = np.concatenate(([sigma0,], sigma, [sigman,]))

    # Construct interpolation function
    _interp = interp1d(nu, sigma, kind='linear', bounds_error=False)
    return lambda freq: np.exp(_interp(freq))


def generate_data(nu, fsigma_T, fsigma_P, components,
                  noise_file="data/core_plus_extended_noise.dat",
                  idx_px = 0):
    """
    Create a mock data vector from a given set of models, including adding a
    noise realization.
    """
    # Loop over components that were included in the data model and calculate
    # the signal at a given frequency (should be in uK_CMB)
    signal = 0
    cmb_signal = 0
    # Disabled for the case of the allsky
    if idx_px == 0:
        pass #print( "(FITTING.PY) Parameters in the input model:" )
    for comp in components:
        if idx_px == 0:
            pass #print comp.param_names

        # Add this component to total signal
        signal += np.atleast_2d(comp.amps()).T * comp.scaling(nu)

        # Store CMB signal separately
        if comp.model == 'cmb':
            cmb_signal = np.atleast_2d(comp.amps()).T * comp.scaling(nu)

    # Construct data vector
    D_vec = np.matrix(signal.flatten()).T

    # Noise rms as a function of frequency
    sigma_interp = noise_model(fname=noise_file, scale=1.)
    sigma_nu = sigma_interp(nu / 1e9)
    fsigma = np.zeros(3*len(nu))
    fsigma[0:len(nu)] = fsigma_T * sigma_nu # Stokes I
    fsigma[len(nu):2*len(nu)] = fsigma_P * sigma_nu # Stokes Q
    fsigma[2*len(nu):] = fsigma_P * sigma_nu # Stokes U

    #noise_mat = np.matrix( np.diagflat(cmb_signal.flatten() * fsigma) )
    #noise_mat = np.matrix( np.diagflat(fsigma) )
    #Ninv = np.linalg.inv(noise_mat)

    # Inverse noise covariance
    noise_mat = np.identity(fsigma.size) * fsigma
    Ninv = np.identity(fsigma.size) / fsigma**2.
    n_vec = (np.matrix(np.random.randn(D_vec.size)) * noise_mat).T

    # Add noise to generated data
    D_vec += n_vec
    return D_vec, Ninv


def model_test(nu, D_vec, Ninv, models_fit, initial_vals=None, burn=500,
               steps=1000, cmb_amp_in=None, sample_file=None):
    """
    Generate simulated data given an input model, and perform MCMC fit using
    another model.
    """
    # Collect together data and noise/instrument model
    Ninv_sqrt = np.matrix(sqrtm(Ninv)) # Invert noise covariance matrix
    beam_mat = np.identity(3*len(nu)) # Beam model
    data_spec = (nu, D_vec, Ninv, beam_mat)

    # Loop over specified component models and set up MCMC parameters for them
    pnames = []; pvals = []; parent_model = []
    for mod in models_fit:
        # Get parameter names, initial parameter values, and component ID
        pn = mod.param_names
        pv = mod.params()

        # Loop through parameters from this component
        for i in range(len(pn)):
            pnames.append( "%s.%s" % (mod.name, pn[i]) )
            pvals.append( pv[i] )
            parent_model.append( mod )

    # Use 'guess' as the initial point for the MCMC if specified
    if initial_vals is None: initial_vals = pvals

    # Collect names, initial values, and parent components for the parameters
    param_spec = (pnames, initial_vals, parent_model)

    # Run MCMC sampler on this model
    t0 = time.time()
    params_out, pnames, samples = mcmc(data_spec, models_fit, param_spec,
                                       burn=burn, steps=steps,
                                       sample_file=sample_file)
    print "MCMC run in %d sec." % (time.time() - t0)

    # Estimate error on recovered CMB amplitudes
    # FIXME: Why estimate error using F_matrix on median!?
    F_fg, F_cmb, F = F_matrix(params_out, nu, models_fit, param_spec)

    H = F_fg.T * Ninv * F_fg

    # Equation A3
    x_mat = np.linalg.inv(F.T * beam_mat.T * Ninv * beam_mat * F) \
          * F.T * beam_mat.T * Ninv * D_vec

    # Equation A14
    U, Lambda, VT = np.linalg.svd(Ninv_sqrt*F_fg, full_matrices=False)

    # Equation A16
    N_eff_inv_cmb = F_cmb.T * Ninv_sqrt \
                  * (np.matrix(np.identity(U.shape[0])) - U*U.T) \
                  * Ninv_sqrt * F_cmb

    N_eff_cmb = np.linalg.inv(N_eff_inv_cmb)
    cmb_noise = np.array([N_eff_cmb[0,0], N_eff_cmb[1,1], N_eff_cmb[2,2]])

    gls_cmb = x_mat[0:3,0]
    cmb_chisq = (np.matrix(cmb_amp_in).T - gls_cmb).T * N_eff_inv_cmb \
              * (np.matrix(cmb_amp_in).T - gls_cmb)

    return gls_cmb, cmb_chisq, cmb_noise
