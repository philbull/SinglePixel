import models
import model_values_allsky
from utils import rj2cmb

# Define input models and their amplitudes/parameters
## Naming conventions follow those in model_list.py

## CMB model
def cmb_model( CMB_I = 50, CMB_Q = 0.6, CMB_U = 0.6 ):
    cmb_model = models.CMB( amp_I = CMB_I, amp_Q = CMB_Q, amp_U = CMB_U )
    return cmb_model

## Synchrotron model
def sync_model( SYNC_I = 30, SYNC_Q = 10, SYNC_U = 10, SYNC_BETA = -3.2 ):
    sync_model = models.SyncPow( amp_I = SYNC_I, amp_Q = SYNC_Q, amp_U = SYNC_U, sync_beta = SYNC_BETA )
    return sync_model

## Free-free model
def ff_model( FF_I = 30, FF_Q = 0, FF_U = 0, FF_BETA = -0.118 ):
    ff_model = models.FreeFreeUnpol( amp_I = rj2cmb( 30e9, FF_I ),
                                     amp_Q = rj2cmb( 30e9, FF_Q ),
                                     amp_U = rj2cmb( 30e9, FF_U ),
                                     ff_beta = FF_BETA )
    return ff_model

## Dust models

def dust_model( DUST_I = 50, DUST_Q = 10 / 1.41, DUST_U = 10 / 1.41,
                DUST_BETA = 1.6, DUST_T = 20. ):
    dust_model = models.DustMBB( amp_I=rj2cmb( 353e9, DUST_I),
                                 amp_Q=rj2cmb( 353e9, DUST_Q),
                                 amp_U=rj2cmb( 353e9, DUST_U),
                                 dust_beta = DUST_BETA, 
                                 dust_T = DUST_T )
    return dust_model

# Silicate + Carbonaceous grains as 2 MBBs
def two_comp_silcar_model( DUST_I = 50, DUST_Q = 10 / 1.41, DUST_U = 10 / 1.41, 
                           DUST_BETA = 1.6, DUST_DBETA = 0.2,
                           DUST_T1 = 18.,
                           DUST_T2 = 22.,
                           DUST_fI = 0.25, DUST_fQ = 0.25, DUST_fU = 0.25 ):
    two_comp_silcar_model = models.DustGen( amp_I=rj2cmb( 353e9, DUST_I ) , #/ ( 1 + DUST_fI ) ), 
                                            amp_Q=rj2cmb( 353e9, DUST_Q ) , #/ ( 1 + DUST_fQ ) ),
                                            amp_U=rj2cmb( 353e9, DUST_U ) , #/ ( 1 + DUST_fU ) ),
                                            beta = DUST_BETA,
                                            dbeta = DUST_DBETA,
                                            Td1 = DUST_T1,
                                            Td2 = DUST_T2,
                                            fI = DUST_fI,
                                            fQ = DUST_fQ,
                                            fU = DUST_fU )
    return two_comp_silcar_model

# HD17 Model with Fe
def hd_fe_model( DUST_I = 50, DUST_Q = 10 / 1.41, DUST_U = 10 / 1.41,
                 FCAR_IN = 1.e3, FSILFE_IN = 1.e3, UVAL_IN = 0.0 ):
    hd_fe_model = models.DustHD( amp_I = rj2cmb( 353e9, DUST_I ),
                                 amp_Q = rj2cmb( 353e9, DUST_Q ),
                                 amp_U = rj2cmb( 353e9, DUST_U ),
                                 fcar = FCAR_IN,
                                 fsilfe = FSILFE_IN,
                                 uval = UVAL_IN )
    return hd_fe_model


# Default dictionary with the amplitudes and parameters of foregrounds and CMB
## Same naming convention as in the model_du=ict below
foreground_dict_default = {
    'cmb':		[ 50, 0.6, 0.6 ],
    'sync':		[ 30., 10., 10., -3.2 ],
    'freefree':		[ 30, 0, 0, -0.118 ],
    'mbb':		[ 50., 10 / 1.41, 10 / 1.41, 1.6, 20. ], 
    '2mbb_silcar':	[ 50 / 1.25, 10 / 1.41 / 1.25, 10 / 1.41 / 1.25, 1.6, 0.2, 18., 22., 0.25, 0.25, 0.25 ],
    'hd_fe':		[ 50, 10 / 1.41, 10 / 1.41, 1e3, 1e3, 0. ],
}

# Dictionary of models (see model_values_allsky.py)
def model_dict( fg_dict = foreground_dict_default ):

    model_dict = {
        'cmb':		cmb_model( fg_dict[ 'cmb' ][ 0 ], fg_dict[ 'cmb' ][ 1 ], fg_dict[ 'cmb' ][ 2 ] ), 
        'sync':		sync_model( fg_dict[ 'sync' ][ 0 ], fg_dict[ 'sync' ][ 1 ], fg_dict[ 'sync' ][ 2 ], 
			fg_dict[ 'sync' ][ 3 ] ), 
        'freefree':	ff_model( fg_dict[ 'freefree' ][ 0 ], fg_dict[ 'freefree' ][ 1 ], fg_dict[ 'freefree' ][ 2 ],
			fg_dict[ 'freefree' ][ 3 ] ),
        'mbb':		dust_model( fg_dict[ 'mbb' ][ 0 ], fg_dict[ 'mbb' ][ 1 ], fg_dict[ 'mbb' ][ 2 ], 
			fg_dict[ 'mbb' ][ 3 ], fg_dict[ 'mbb' ][ 4 ] ), 
        '2mbb_silcar':	two_comp_silcar_model( fg_dict[ '2mbb_silcar' ][ 0 ], fg_dict[ '2mbb_silcar' ][ 1 ], fg_dict[ '2mbb_silcar' ][ 2 ], 
			fg_dict[ '2mbb_silcar' ][ 3 ], fg_dict[ '2mbb_silcar' ][ 4 ], fg_dict[ '2mbb_silcar' ][ 5 ],
			fg_dict[ '2mbb_silcar' ][ 6 ], fg_dict[ '2mbb_silcar' ][ 7 ], fg_dict[ '2mbb_silcar' ][ 8 ], 
			fg_dict[ '2mbb_silcar' ][ 9 ] ),
	'hd_fe':	hd_fe_model( fg_dict[ 'hd_fe' ][ 0 ], fg_dict[ 'hd_fe' ][ 1 ], fg_dict[ 'hd_fe' ][ 2 ],
			fg_dict[ 'hd_fe' ][ 3 ], fg_dict[ 'hd_fe' ][ 4 ], fg_dict[ 'hd_fe' ][ 5 ] ),
    }
    return model_dict

def get_model_param_names( in_list = '' , fit_list = '', Nside = 16, Npix = 0 ):

    in_list_all = [ 'cmb', ] ; fit_list_all = [ 'cmb', ]
    # Removing any white spaces and creating a list
    in_list = "".join( in_list.split( ) )
    in_list_all += in_list.split(",")
    fit_list = "".join( fit_list.split( ) )
    fit_list_all += fit_list.split(",")

    # Make sure models are of known types
    allowed_comps = model_dict( fg_dict = model_values_allsky.get_dict(
                    in_list_all, Nside, Npix ) )
    for item in in_list_all:
        if item not in allowed_comps.keys():
            raise ValueError("Unknown component type '%s'" % item)

    for item in fit_list_all:
        if item not in allowed_comps.keys():
            raise ValueError("Unknown component type '%s'" % item)

    # Print recognised models and specify name
    print "Input components:", in_list_all
    print "Fitting components:", fit_list_all

    # Collect components into lists and set input amplitudes
    mods_in = [ allowed_comps[comp] for comp in in_list_all ]
    mods_fit = [ allowed_comps[comp] for comp in fit_list_all ]
    # Parameter names
    amp_names_in = [] ; param_names_in = [] ;
    amp_names_fit = [] ; param_names_fit = [] ;
    for mod in mods_in:
        amp_names_in += ["%s_%s" % (mod.model, pol) for pol in "IQU"]
        param_names_in += mod.param_names
    for mod in mods_fit:
        amp_names_fit += ["%s_%s" % (mod.model, pol) for pol in "IQU"]
        param_names_fit += mod.param_names
    name_in = "-".join( in_list_all )
    name_fit = "-".join( fit_list_all )
    return amp_names_in, param_names_in, amp_names_fit, param_names_fit, name_in, name_fit
