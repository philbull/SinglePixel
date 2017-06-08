# Function to get the amplitudes and parameters for the foreground models. 
## See repository2data.py for details about the units and reference frequencies of the input data read below.
## Model naming follows model_list_allsky
## Amplitude and parameter order follows model_list_allsky

import models as model_single_pixel
import healpy as hp
import numpy as np

import sys, platform, os
print('Using CAMB installed at '+ os.path.realpath(os.path.join(os.getcwd(),'..')))
sys.path.insert(0,os.path.realpath(os.path.join(os.getcwd(),'..')))
import camb
from camb import model, initialpower

import pdb
import matplotlib.pyplot as plt

def get_cmb( Nside, Npix ):
    pars = camb.CAMBparams()
    #This function sets up CosmoMC-like settings, with one massive neutrino and helium set using BBN consistency
    pars.set_cosmology( H0 = 67.74, ombh2 = 0.0223, omch2 = 0.1188, omk = 0, tau = 0.066 ) ;
    # Tensors on
    pars.WantTensors = True
    pars.InitPower.set_params( ns = 0.9667, r = 0.5 ) ;
    # Power spectrum will be 3 * Nside - 1
    pars.set_for_lmax( 3 * Nside - 1, lens_potential_accuracy=0);
    #calculate results for these parameters
    results = camb.get_results(pars) ;
    #get dictionary of CAMB power spectra
    cl_total = results.get_total_cls( 3 * Nside - 1 )
    # Needs to be re-arranged
    tmp = [ cl_total[ :, 0 ] , cl_total[ :, 1 ], cl_total[ :, 2 ], cl_total[ :, 3 ] ]
    cmb_synfast = hp.synfast( tmp, Nside, new = True )
    # K->micro K!
    cmb_out = [ cmb_synfast[ 0 ][ Npix ] * 1e6, cmb_synfast[ 1 ][ Npix ] * 1e6, cmb_synfast[ 2 ][ Npix ] * 1e6 ]
    return cmb_out

def get_sync( Nside, Npix ):
    # Opening the corresponding files
    sync_all = hp.read_map( "data/sync_IQU_%04d.fits" % Nside, np.arange( 3 ), verbose = False )
    sync_out = [ sync_all[ 0 ][ Npix ], sync_all[ 1 ][ Npix ], sync_all[ 2 ][ Npix ] ]
    beta_sync_all = hp.read_map( "data/sync_beta_%04d.fits" % Nside, verbose = False )
    sync_out.append( beta_sync_all[ Npix ] )
    return sync_out

def get_freefree( Nside, Npix ):
    # Opening the corresponding file
    ff_all = hp.read_map( "data/freefree_I_%04d.fits" % Nside, np.arange( 1 ), verbose = False )
    # Recall free-free is set as non polarized
    ff_out = [ ff_all[ Npix ], 0, 0 ]
    # recall beta_freefree is set as constant over the sky for now
    ff_out.append( -0.118 )
    return ff_out


def get_mbb( Nside, Npix ):
    # Opening the corresponding files
    dust_all = hp.read_map( "data/dust_IQU_%04d.fits" % Nside, np.arange( 3 ), verbose = False )
    dust_out = [ dust_all[ 0 ][ Npix ], dust_all[ 1 ][ Npix ], dust_all[ 2 ][ Npix ] ]
    beta_dust_all = hp.read_map( "data/dust_beta_%04d.fits" % Nside, verbose = False )
    dust_out.append( beta_dust_all[ Npix ] )
    temp_dust_all = hp.read_map( "data/dust_temp_%04d.fits" % Nside, verbose = False )
    dust_out.append( temp_dust_all[ Npix ] )
    return dust_out

def get_2mbb_silcar( Nside, Npix ):
    # Opening the corresponding files
    dust_all = hp.read_map( "data/dust_IQU_%04d.fits" % Nside, np.arange( 3 ), verbose = False )
    dust_out = [ dust_all[ 0 ][ Npix ], dust_all[ 1 ][ Npix ], dust_all[ 2 ][ Npix ] ]
    beta_dust_all = hp.read_map( "data/dust_beta_%04d.fits" % Nside, verbose = False )
    dust_out.append( beta_dust_all[ Npix ] )
    #PS: The next calls are thought for a single pixel, but can be made an array and sent to each pixel process, once the code is running full MPI
    np.random.seed( Npix )
    # Delta beta (for now uniform distribution between two values )
    dust_out.append( beta_dust_all[ Npix ] + np.random.uniform( 0.2, 0.5, 1 ) )
    # Dust temperature 1
    temp_dust_all = hp.read_map( "data/dust_temp_%04d.fits" % Nside, verbose = False )
    dust_out.append( temp_dust_all[ Npix ] )
    # Dust temperatue 2 (for now uniform distribution between two values)
    dust_out.append( temp_dust_all[ Npix ] + np.random.uniform( 3, 6, 1 ) )
    # Relative fractioni of I of the second dust component (fI)
    dust_out.append( np.random.uniform( 0.2, 0.4, 1 ) ) 
    # Relative fraction of Q of the second dust component (fQ)
    dust_out.append( np.random.uniform( 0.2, 0.4, 1 ) )
    # Relative fraction of U of the second dust component (fU)
    dust_out.append( np.random.uniform( 0.2, 0.4, 1 ) )
    return dust_out


# NB: HD models are normalized at Planck 353 GHz dust maps.
def get_hd_fe( Nside, Npix ):
    # Opening the corresponding files
    dust_all = hp.read_map( "data/dust_IQU_%04d.fits" % Nside, np.arange( 3 ), verbose = False )
    hd_fe_out = [ dust_all[ 0 ][ Npix ], dust_all[ 1 ][ Npix ], dust_all[ 2 ][ Npix ] ]
    beta_dust_all = hp.read_map( "data/dust_beta_%04d.fits" % Nside, verbose = False )
    temp_dust_all = hp.read_map( "data/dust_temp_%04d.fits" % Nside, verbose = False )

    # From Brandon Hensley
    # Draw a log U value based on the Commander dust temperature and beta
    # U \propto T^(4 + beta)
    # Note: data files currently only for -0.5 < log U < 0.5 (FIXME: Sergi: it seems this is not any more the case, given the limits set to -3, 5 below
    uval_out = ( 4. + beta_dust_all[ Npix ] ) * np.log10( temp_dust_all[ Npix ] / np.mean( temp_dust_all ) )
    if uval_out < -3.0:
        uval_out = -3.0
    if uval_out > 5.0:
        uval_out = 5.0

    # Allow dust amplitudes to vary stochastically by 10%
    ## Repeatability
    np.random.seed( Npix )
    fcar_draw = np.random.randn()*0.1 + 1.
    if fcar_draw < 0.:
        fcar_draw = 0.
    fsilfe_draw = np.random.rand()

    # Allow varying amounts of Fe
    fsil_fac = 1. / ( 1. - fsilfe_draw )
    fcar_out = fcar_draw * fsil_fac
    fsilfe_out = fsilfe_draw * fsil_fac
     
    # Appending in the order the parameters are read by the dictionary in model_list_allsky
    hd_fe_out.append( fcar_out )
    hd_fe_out.append( fsilfe_out )
    hd_fe_out.append( uval_out )

    return hd_fe_out

# Dictionary to be used in run_joint_mcmc_allsky
def get_dict( in_list, Nside, Npix ):

    foreground_dict = {
        'cmb':			get_cmb( Nside, Npix ),
        'sync':			get_sync( Nside, Npix ),
        'freefree':		get_freefree( Nside, Npix ),
        'mbb':			get_mbb( Nside, Npix ),
        '2mbb_silcar':		get_2mbb_silcar( Nside, Npix ),
        'hd_fe':		get_hd_fe( Nside, Npix ),
    }
    return foreground_dict




