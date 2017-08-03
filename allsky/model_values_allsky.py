# Function to get the amplitudes and parameters for the foreground models. 
## See repository2data.py for details about the units and reference frequencies of the input data read below.
## Model naming follows model_list_allsky
## Amplitude and parameter order follows model_list_allsky

import healpy as hp
import numpy as np

import sys, platform, os
#print('Using CAMB installed at '+ os.path.realpath(os.path.join(os.getcwd(),'..')))
#sys.path.insert(0,os.path.realpath(os.path.join(os.getcwd(),'..')))
import camb
from camb import model, initialpower

import pdb
import matplotlib.pyplot as plt

def get_cmb( Nside, px_unseen ):
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
    # This comes out as a list
    cmb_synfast = hp.synfast( tmp, Nside, new = True )
    # Rather cast it into a numpy array and change K->micro K!
    cmb_out = np.zeros( ( 3, len( px_unseen ) ), dtype = 'float32' )
    cmb_out[ 0 ] = cmb_synfast[ 0 ][ px_unseen ] * 1e6
    cmb_out[ 1 ] = cmb_synfast[ 1 ][ px_unseen ] * 1e6
    cmb_out[ 2 ] = cmb_synfast[ 2 ][ px_unseen ] * 1e6
    return cmb_out

def get_sync( Nside, px_unseen ):
    # Opening the corresponding files
    # This comes out as a list
    sync_all = hp.read_map( "data/sync_IQU_%04d.fits" % Nside, np.arange( 3 ), verbose = False )
    # Rather cast it into a numpy array
    sync_out = np.zeros( ( 4, len( px_unseen ) ), dtype = 'float32' )
    sync_out[ 0 ] = sync_all[ 0 ][ px_unseen ] 
    sync_out[ 1 ] = sync_all[ 1 ][ px_unseen ]
    sync_out[ 2 ] = sync_all[ 2 ][ px_unseen ]
    beta_sync_all = hp.read_map( "data/sync_beta_%04d.fits" % Nside, verbose = False )
    sync_out[ 3 ] = beta_sync_all[ px_unseen ]
    return sync_out

def get_freefree( Nside, px_unseen ):
    # Opening the corresponding file
    # This comes out as a list
    ff_all = hp.read_map( "data/freefree_I_%04d.fits" % Nside, np.arange( 1 ), verbose = False )
    # Rather cast it into a numpy array
    ff_out = np.zeros( ( 4, len( px_unseen ) ), dtype = 'float32' )
    # Recall free-free is set as non polarized
    ff_out[ 0 ] = ff_all[ px_unseen ]
    # recall beta_freefree is set as constant over the sky for now
    ff_out[ 3 ] = -0.118 * np.ones( len( px_unseen ), dtype = 'float32' ) 
    return ff_out


def get_mbb( Nside, px_unseen ):
    # Opening the corresponding files
    # This comes out as a list
    dust_all = hp.read_map( "data/dust_IQU_%04d.fits" % Nside, np.arange( 3 ), verbose = False )
    # Rather cast it into a numpy array
    dust_out = np.zeros( ( 5, len( px_unseen ) ), dtype = 'float32' )
    dust_out[ 0 ] = dust_all[ 0 ][ px_unseen ]
    dust_out[ 1 ] = dust_all[ 1 ][ px_unseen ]
    dust_out[ 2 ] = dust_all[ 2 ][ px_unseen ]
    beta_dust_all = hp.read_map( "data/dust_beta_%04d.fits" % Nside, verbose = False )
    dust_out[ 3 ] = beta_dust_all[ px_unseen ]
    temp_dust_all = hp.read_map( "data/dust_temp_%04d.fits" % Nside, verbose = False )
    dust_out[ 4 ] = temp_dust_all[ px_unseen ]
    return dust_out

def get_2mbb_silcar( Nside, px_unseen, seed ):
    # Opening the corresponding files
    # This comes out as a list
    dust_all = hp.read_map( "data/dust_IQU_%04d.fits" % Nside, np.arange( 3 ), verbose = False )
    # Rather cast it into a numpy array
    dust_out = np.zeros( ( 10, len( px_unseen ) ), dtype = 'float32' )
    dust_out[ 0 ] = dust_all[ 0 ][ px_unseen ]
    dust_out[ 1 ] = dust_all[ 1 ][ px_unseen ]
    dust_out[ 2 ] = dust_all[ 2 ][ px_unseen ]
    beta_dust_all = hp.read_map( "data/dust_beta_%04d.fits" % Nside, verbose = False )
    dust_out[ 3 ] = beta_dust_all[ px_unseen ]
    # Random seed
    np.random.seed( seed )
    # Delta beta (for now uniform distribution between two values )
    dust_out[ 4 ] = beta_dust_all[ px_unseen ] + np.random.uniform( 0.2, 0.5, len( px_unseen ) )
    # Dust temperature 1
    temp_dust_all = hp.read_map( "data/dust_temp_%04d.fits" % Nside, verbose = False )
    dust_out[ 5 ] = temp_dust_all[ px_unseen ]
    # Dust temperatue 2 (for now uniform distribution between two values)
    dust_out[ 6 ] = temp_dust_all[ px_unseen ] + np.random.uniform( 3, 6, len( px_unseen ) )
    # Relative fractioni of I of the second dust component (fI)
    dust_out[ 7 ] = np.random.uniform( 0.2, 0.4, len( px_unseen ) )
    # Relative fraction of Q of the second dust component (fQ)
    dust_out[ 8 ] = np.random.uniform( 0.2, 0.4, len( px_unseen ) )
    # Relative fraction of U of the second dust component (fU)
    dust_out[ 9 ] = np.random.uniform( 0.2, 0.4, len( px_unseen ) )
    return dust_out


# NB: HD models are normalized at Planck 353 GHz dust maps.
def get_hd_fe( Nside, px_unseen, seed ):
    # Opening the corresponding files
    # This comes out as a list
    dust_all = hp.read_map( "data/dust_IQU_%04d.fits" % Nside, np.arange( 3 ), verbose = False )
    # Rather cast it into a numpy array
    hd_fe_out = np.zeros( ( 6, len( px_unseen ) ), dtype = 'float32' )
    hd_fe_out[ 0 ] = dust_all[ 0 ][ px_unseen ]
    hd_fe_out[ 1 ] = dust_all[ 1 ][ px_unseen ]
    hd_fe_out[ 2 ] = dust_all[ 2 ][ px_unseen ]

    beta_dust_all = hp.read_map( "data/dust_beta_%04d.fits" % Nside, verbose = False )
    temp_dust_all = hp.read_map( "data/dust_temp_%04d.fits" % Nside, verbose = False )
    # From Brandon Hensley
    # Draw a log U value based on the Commander dust temperature and beta
    # U \propto T^(4 + beta)
    uval_out = ( 4. + beta_dust_all[ px_unseen ] ) * np.log10( temp_dust_all[ px_unseen ] / np.mean( temp_dust_all ) )
    idx = np.where( uval_out < -3.0 )
    uval_out[ idx ] = -3.0
    idx = np.where( uval_out > 5.0 )
    uval_out[ idx ] = 5.0

    # Allow dust amplitudes to vary stochastically by 10%
    np.random.seed( seed )
    fcar_draw = np.random.randn( len( px_unseen ) ) * 0.1 + 1.
    idx = np.where( fcar_draw < 0. )
    fcar_draw[ idx ] = 0.
    # rand is drawn from [0,1]
    fsilfe_draw = np.random.rand( len( px_unseen ) )

    # Allow varying amounts of Fe
    fsil_fac = 1. / ( 1. - fsilfe_draw )
    fcar_out = fcar_draw * fsil_fac
    fsilfe_out = fsilfe_draw * fsil_fac
     
    # Appending in the order the parameters are read by the dictionary in model_list_allsky
    hd_fe_out[ 3 ] = fcar_out
    hd_fe_out[ 4 ] = fsilfe_out
    hd_fe_out[ 5 ] = uval_out

    return hd_fe_out

# Dictionary to be used in run_joint_mcmc_allsky
def get_allsky_dict_split( in_list, Nside, px_unseen, seed, nproc ):

    foreground_dict = {
        'cmb':			np.array_split( get_cmb( Nside, px_unseen ), nproc, axis = 1 ),
        'sync':			np.array_split( get_sync( Nside, px_unseen ), nproc, axis = 1 ),
        'freefree':		np.array_split( get_freefree( Nside, px_unseen ), nproc, axis = 1 ),
        'mbb':			np.array_split( get_mbb( Nside, px_unseen ), nproc, axis = 1 ),
        '2mbb_silcar':		np.array_split( get_2mbb_silcar( Nside,px_unseen, seed ), nproc, axis = 1 ),
        'hd_fe':		np.array_split( get_hd_fe( Nside, px_unseen, seed ), nproc, axis = 1 ),
    }
    
    # Needs to be reshaped to be scattered by MPI
    dict_tmp = {
        'cmb':		0, 
        'sync':		0,
        'freefree':	0,
        'mbb':		0,
        '2mbb_silcar':	0,
        'hd_fe':	0,
    }

    foreground_dict_list = [] 
    for i_proc in np.arange( nproc ):
        for key in foreground_dict:
            dict_tmp[ key ] = foreground_dict[ key ][ i_proc ]
        # For some reason, appending the dictionary as a whole does not work properly
        foreground_dict_list.append( {
        'cmb':          dict_tmp[ 'cmb' ],
        'sync':         dict_tmp[ 'sync' ],
        'freefree':     dict_tmp[ 'freefree' ],
        'mbb':          dict_tmp[ 'mbb' ],
        '2mbb_silcar':  dict_tmp[ '2mbb_silcar' ],
        'hd_fe':        dict_tmp[ 'hd_fe' ],
    } )
    return foreground_dict_list

