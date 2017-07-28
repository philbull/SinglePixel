import numpy as np
import matplotlib.pyplot as plt
from utils import rj2cmb
from utils import cmb2rj


def run_joint_MCMC(VARIABLE):
    """
    Do MCMC runs to fit FG models to simulated data, over a grid of
    (nu_min, nu_max) values.
    """
    import numpy as np
    import models
    import model_list
    import fitting
    from utils import rj2cmb, bands_log
    import sys, time, os, copy
    #from multiprocessing import Pool
    #from mpi4py import MPI

    # Set-up MPI
    #comm = MPI.COMM_WORLD
    #myid = comm.Get_rank()
    #nproc = comm.Get_size()

    # Prefix for output files
    PREFIX = "final"
    NBURN = 50
    NSTEPS = 100
    NWALKERS = 50

    # Reference noise curve (assumes noise file contains sigma_P=sigma_Q=sigma_U
    # in uK_CMB.deg for a given experiment)
    NOISE_FILE = "data/noise_coreplus_extended.dat"
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

    # Define input models and their amplitudes/parameters
    allowed_comps = model_list.model_dict
    #Commented out cmb_model = model_list.cmb_model


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Temporarily adding this code

    cmb_model = models.CMB( amp_I=50., amp_Q=0.6, amp_U=0.6 )


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



    # Parse args to define input and output models
    if len(sys.argv) > 2:
        in_list += sys.argv[2].split(",")
        fit_list += sys.argv[3].split(",")
    else:
        in_list = ['cmb', 'synch', '2mbb_cloud']
        fit_list = ['cmb', 'synch', 'genmbb']

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

    # Frequency ranges
    numin_vals = [30.]
    numax_vals = [500.]

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
    filename = "output/%s_summary_%s.%s_nb%d_seed%d" \
                 % (PREFIX, name_in, name_fit, nbands, SEED)
    cut_range = np.arange(NSTEPS, step=200)
    for cut in cut_range:
        fname = filename + "_cut%d.dat" % cut
        f = open(fname, 'w')
        f.close()


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


    def run_model(nu_params):
        # Get band definition
        nu_min, nu_max = nu_params
        print "nu_min = %d GHz, nu_max = %d GHz" % (nu_min, nu_max)
        nu = bands_log(nu_min, nu_max, nbands)
        label = str(nu_min) + '_' + str(nu_max)

        # Make copies of models
        my_mods_in = mods_in
        my_mods_fit = mods_fit

        # Name of sample file
        fname_samples = "output/%s_samples_%s.%s_nb%d_seed%d_%s.dat" \
                      % (PREFIX, name_in, name_fit, nbands, SEED, label)
        #fname_samples = None

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

            # Loop over parameter names
            for i in range(len(pnames)):

                # Mean, std. dev., and fractional shift from true value
                _mean = np.mean(samples[i,:,cut:])
                _std = np.std(samples[i,:,cut:])
                _fracbias = (np.mean(samples[i,:,cut:]) - ini[i]) \
                          / np.std(samples[i,:,cut:])
                stats = [_mean, _std, _fracbias]

                # Keep summary stats, to be written to file
                summary_data += stats
                header += "mean_%s std_%s Delta_%s " \
                        % (pnames[i], pnames[i], pnames[i])

                print "%14s: %+3.3e +/- %3.3e [Delta = %+3.3f]" \
                      % (pnames[i], stats[0], stats[1], stats[2])

            # If file is empty, set flag to write header when saving output
            has_header = False if os.stat(fname).st_size == 0 else True

            # Append summary statistics to file
            f = open(fname, 'a')
            if has_header:
                np.savetxt(f, np.atleast_2d(summary_data))
            else:
                np.savetxt(f, np.atleast_2d(summary_data), header=header[:-1])
            f.close()

    # Run pool of processes
    #pool = Pool(NPROC)
    #pool.map(run_model, nu_params)

    for i in range(len(nu_params)):
    #    if i % nproc != myid: continue

        # Run the model for this set of params
        run_model(nu_params[i])


def bias_calc():
    spectral_data = np.loadtxt('output/final_samples_cmb-synch-2mbb_cloud.cmb-synch-genmbb_nb7_seed10_30.0_500.0.dat')
    cmb_I_Chain_Values = spectral_data.T[0]
    cmb_Q_Chain_Values = spectral_data.T[1]
    cmb_U_Chain_Values = spectral_data.T[2]

    summary_stat_data = np.loadtxt('output/final_summary_cmb-synch-2mbb_cloud.cmb-synch-genmbb_nb7_seed10_cut0.dat')
    N_experiments = len(cmb_I_Chain_Values)

    #Indices of [mean_cmb_I, std_cmb_I, mean_cmb_q, std_cmb_q, mean_cmb_U, std_cmb_U] in Summary Array
    summary_stat_data = np.take(summary_stat_data,[3,4,6,7,9,10])


    cmb_I_inputed = 50
    cmb_Q_inputed = 0.6
    cmb_U_inputed = 0.6

    cmb_I_bias = (summary_stat_data[0] - cmb_I_inputed )/ summary_stat_data[1]
    cmb_Q_bias = (summary_stat_data[2] - cmb_Q_inputed )/ summary_stat_data[3]
    cmb_U_bias = (summary_stat_data[4] - cmb_U_inputed )/ summary_stat_data[5]


    plt.suptitle("Bias in Recovery of Stokes Parameters using SinglePixel  with " + str(N_experiments) + " Experiments" +
                "\n" + "")

    plt.subplot(221)
    plt.title("I$_{CMB}$ Experiment")
    plt.axvline(color = "red", x = cmb_I_inputed)
    plt.hist(cmb_I_Chain_Values, bins = 30)
    plt.xlabel("Predicted I$_{CMB}$ in $\mu$K$_{RJ}$")
    plt.ylabel("N")



    plt.subplot(222)
    plt.title("Q$_{CMB}$ Experiment")
    plt.axvline(color = "red", x = cmb_Q_inputed)
    plt.hist(cmb_Q_Chain_Values, bins = 30)
    plt.xlabel("Predicted Q$_{CMB}$ in $\mu$K$_{RJ}$")
    plt.ylabel("N")

    plt.subplot(223)
    plt.title("U$_{CMB}$ Experiment")
    plt.axvline(color = "red", x = cmb_U_inputed)
    plt.hist(cmb_U_Chain_Values, bins = 30)
    plt.xlabel("Predicted U$_{CMB}$ in $\mu$K$_{RJ}$")
    plt.ylabel("N")

    plt.subplot(224)
    plt.axhline(color = "black", y= 0)
    plt.axis([-1,3,-1,1])
    objects = ('I$_{CMB}$', 'Q$_{CMB}$', 'U$_{CMB}$')
    y_pos = np.arange(len(objects))
    bias_array = [cmb_I_bias, cmb_Q_bias, cmb_U_bias]
    plt.bar(y_pos, bias_array, align='center', alpha=1.0)
    plt.xticks(y_pos, objects)
    plt.ylabel('Bias Units?')
    plt.title("Bias in Recovered Polarization Parameters")

    plt.tight_layout()
    plt.show()

for i in np.arange(1,5):
    run_joint_MCMC(5)
    bias_calc()