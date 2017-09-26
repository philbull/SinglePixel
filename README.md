# SinglePixel

SinglePixel is a code for modelling, simulating, and fitting the frequency spectra of complex dust foregrounds, as well as other components of the microwave sky. It operates on individual pixels (i.e. single lines of sight), hence the name.

## Code

The code is written entirely in Python, and makes use of the <tt>numpy</tt>, <tt>scipy</tt>, <tt>matplotlib</tt>, <tt>mpi4py</tt>, and <tt>emcee</tt> packages. You can clone it from the <b><a href="https://github.com/philbull/SinglePixel">philbull/SinglePixel</a></b> repository on GitHub. It is available under an open source MIT license, and contributions (e.g. pull requests and bug reports) are welcome. Please cite the paper (arXiv:1709.07897) if you use or modify the code.</p>

The following is a list of the scripts that make up SinglePixel:
 * <tt>models.py</tt>: Python classes that define the various dust models and other foregrounds.
 * <tt>model_list.py</tt>: Instantiations of the various dust/foreground model classes with representative parameter values.
 * <tt>fitting.py</tt>: Infrastructure for simulating and fitting multi-band frequency spectra.
 * <tt>run_joint_mcmc.py</tt>: Main script for running the simulation/fitting procedure over a grid of models, band specifications, and noise realisations. MPI-enabled.

If you wish to repeat the analysis in the paper, you will find everything you need in run_joint_mcmc.py. Please study the code carefully before running, as the script must be modified in order to set certain parameters. Importantly, if you do not want to generate large amounts of output by storing the raw MCMC samples to disk, make sure that you have set <tt>fname_samples = None</tt> in the <tt>model_test()</tt> function.

The <tt>run_joint_mcmc.py</tt> script accepts several command line arguments that specify the models to simulate, the models to fit, and the random seed to use. Example usage:

    $ mpirun -n 2 python run_joint_mcmc.py 99 synch,2mbb_fe synch,genmbb

This fits a CMB + synchrotron + 2MBB model to a CMB + synchrotron + Fe input model, with a random seed value of 99. Note that there are no spaces between the commas.

## Data

See http://philbull.com/singlepixel/ for downloadable datafiles that were generated using this code as part of our study.
