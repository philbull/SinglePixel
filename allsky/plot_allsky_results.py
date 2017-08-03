# Program to view the results of an allsky run

import model_list_allsky
#import model_values_allsky
import healpy as hp
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as grd
import numpy as np

import pdb

def main( in_list = 'sync,freefree,2mbb_silcar',
          fit_list = 'sync,freefree,2mbb_silcar',
          Nside = 32,
          seed = 100 ):

    # Getting the relevant strings
    amp_names_in, param_names_in, amp_names_fit, param_names_fit, name_in, name_fit = \
    model_list_allsky.get_model_param_names( in_list = in_list, fit_list = fit_list )

    filename = 'output/allsky_summary_%s.%s_nb7_seed%04i_nside%04i.fits' % ( name_in, name_fit, seed, Nside )
    print "Reading the results from %s" % filename
    # out*3: initial/final/difference
    N_fit = len( amp_names_fit ) + len( param_names_fit )
    Nmaps = N_fit * 3
    maps = hp.read_map( filename, range( Nmaps ) )
    
    # Allsky maps
    # For now, needs to get amplitude names and parameter names. Check reference frequencies and units, as well.
    ttl_plt = amp_names_fit + param_names_fit 
    unts_plt = [ 'micro K CMB', 'micro K CMB', 'micro K CMB', 'micro K RJ', 'micro K RJ', 'micro K RJ', 'micro K RJ', 'micro K RJ', 'micro K RJ', '', '', 'K' ]
    # Temporary
    if ( Nmaps / 3 ) != len( unts_plt ):
        unts_plt = [ "" for x in range( Nmaps / 3 ) ]
  
    for i_mp in range( Nmaps / 3 ):
        fig = plt.figure( num = None, figsize=( 12, 8 ), dpi=80, facecolor='w', edgecolor='k')
        # Initial. Discarding totally odd values, or UNSEEN, for the computation of the plot range
        q_gd = np.where( np.abs( maps[ i_mp * 3 ] ) < 1e10 ) 
        # sgm = 2
        mn_mllvw = np.percentile( maps[ i_mp * 3 ][ q_gd ], 5 ) # np.min( maps[ i_mp * 3 ][ q_gd ] ) - sgm * np.std( maps[ i_mp * 3 ][ q_gd ] )
        mx_mllvw = np.percentile( maps[ i_mp * 3 ][ q_gd ], 95 ) # np.max( maps[ i_mp * 3 ][ q_gd ] ) + sgm * np.std( maps[ i_mp * 3 ][ q_gd ] )

        hp.mollview( maps[ i_mp * 3 ], norm = 'linear', sub = ( 2, 2, 1 ), title = 'INITIAL MAP', 
                     unit = unts_plt[ i_mp ], min = mn_mllvw, max = mx_mllvw )
        # Output
        hp.mollview( maps[ i_mp * 3 + 1 ], norm = 'linear', sub = ( 2, 2, 2 ), title = 'FINAL MAP', 
                     unit = unts_plt[ i_mp ], min = mn_mllvw, max = mx_mllvw )

        # Bias in terms of the RMS
        frc_bias = np.full( 12 * Nside * Nside, hp.UNSEEN )
        # Only fill in values with pixels that had non-zero error values
        q_gd_2 = np.where( maps[ i_mp * 3 + 2 ] != 0 )
        ## Tuple to array
        q_gd_2 = q_gd_2[ 0 ] 
        frc_bias[ q_gd_2 ] = ( maps[ i_mp * 3 + 1 ][ q_gd_2 ] - maps[ i_mp * 3 ][ q_gd_2 ] ) / maps[ i_mp * 3 + 2 ][ q_gd_2 ]
        mn_mllvw = np.percentile( frc_bias[ q_gd ], 10 ) # np.min( frc_bias[ q_gd ] )
        mx_mllvw = np.percentile( frc_bias[ q_gd ], 90 ) # np.max( frc_bias[ q_gd ] )
        if mn_mllvw == mx_mllvw:
            mn_mllvw = 0
            mx_mllvw = 0

        hp.mollview( frc_bias, norm = 'linear', sub = ( 2, 1, 2 ), title = 'FRACTIONAL BIAS', 
                     margins = [ 0, 0.02, 0, 0.02 ], 
                     unit = 'x 1 sigma', min = mn_mllvw, max = mx_mllvw )
        plt.suptitle( ttl_plt[ i_mp ], fontsize = 18 )
        nm_fig = 'fig/final_summary_%s.%s_nb7_seed100_nside%04i_%s.png' % ( name_in, name_fit, Nside, ttl_plt[ i_mp ] )
        fig.savefig( nm_fig )
    
    plt.show() 

    # Power spectrum
    cmb_maps_in = [ maps[ 0 ], maps[ 3 ], maps[ 6 ] ]
    cl_in = hp.anafast( cmb_maps_in, lmax = 3 * Nside - 1 )
    # Noise realizations
    N_NS = 100
    # Loop over (array of normal distributed values, different at each pixel)
    cl_tmp = np.zeros( ( N_NS, 6, 3 * Nside ), dtype = 'float32' )
    # Number of pixels in the map
    n_pix_map = 12 * Nside * Nside
    # If there are too many UNSEEN values, skip this part
    q_unseen = ( maps[ 0 ] == hp.UNSEEN )
    for i_ns in range( N_NS ):
        if sum( q_unseen ) < 0.03 * 12 * Nside * Nside:
            cmb_tmp = [ maps[ 1 ] + np.random.normal( 0, 1, n_pix_map ) * maps[ 2 ],
                        maps[ 4 ] + np.random.normal( 0, 1, n_pix_map ) * maps[ 5 ] ,
                        maps[ 7 ] + np.random.normal( 0, 1, n_pix_map ) * maps[ 8 ] ]
            cl_tmp[ i_ns, : ] = hp.anafast( cmb_tmp, lmax = 3 * Nside - 1 )
        else:
            cl_tmp[ i_ns, : ] = cl_in

    # Mean and std per bin, plot 1, 2 3 sigma.
    cl_plt = np.zeros( ( 6, 3 * Nside, 2 ), dtype = 'float32' )
    cl_plt[ :, :, 0 ] = np.mean( cl_tmp, axis = 0 )
    cl_plt[ :, :, 1 ] = np.std( cl_tmp, axis = 0 )
    ell = np.arange( 3 * Nside )
    fct_ell = ell * ( ell + 1 ) / 2 / np.pi
    fct_ell[ : ] = 1
    # Some plots
    # Shades from https://stackoverflow.com/questions/25994048/confidence-regions-of-1sigma-for-a-2d-plot
    ttl_plt = [ 'TT', 'EE', 'BB', 'TE', 'TB', 'EB' ]
    for i_plt in range( 6 ):
        fig = plt.figure( num = None, figsize=( 9, 6 ), dpi=80, facecolor='w', edgecolor='k')
        if i_plt < 3:
            plt.loglog( ell, fct_ell * cl_in[ i_plt ], '-k', marker = 'o', label = 'Initial' )
            plt.loglog( ell, fct_ell * cl_plt[ i_plt, :, 0 ], '-b', marker = 'o', label = 'Final' )
            # Just avoiding to overplot a fit that is the initial
            if sum( q_unseen ) > 0.03 * 12 * Nside * Nside:
                plt.loglog( ell, fct_ell * cl_in[ i_plt ], '-k', marker = 'o' )
        else:
            plt.plot( ell, fct_ell * cl_in[ i_plt ], '-k', marker = 'o', label = 'Initial' )
            plt.plot( ell, fct_ell * cl_plt[ i_plt, :, 0 ], '-b', marker = 'o', label = 'Final' )
            # Just avoiding to overplot a fit that is the initial
            if sum( q_unseen ) > 0.03 * 12 * Nside * Nside:
                plt.plot( ell, fct_ell * cl_in[ i_plt ], '-k', marker = 'o' )
        LB3 = fct_ell * ( cl_plt[ i_plt, :, 0 ] - 3 * cl_plt[ i_plt, :, 1 ] )
        UB3 = fct_ell * ( cl_plt[ i_plt, :, 0 ] + 3 * cl_plt[ i_plt, :, 1 ] )
        plt.fill_between( ell, LB3, UB3, where = UB3 >= LB3, facecolor='blue', alpha= 0.1, zorder = 0 )
        LB2 = fct_ell * ( cl_plt[ i_plt, :, 0 ] - 2 * cl_plt[ i_plt, :, 1 ] )
        UB2 = fct_ell * ( cl_plt[ i_plt, :, 0 ] + 2 * cl_plt[ i_plt, :, 1 ] )
        plt.fill_between( ell, LB2, UB2, where = UB2 >= LB2, facecolor='blue', alpha= 0.2, zorder = 0 )
        #plt.loglog( ell, LB2, '-.b', zorder = 1, alpha = 0.3 )
        #plt.loglog( ell, UB2, '-.b', zorder = 1, alpha = 0.3 )
        LB = fct_ell * ( cl_plt[ i_plt, :, 0 ] - cl_plt[ i_plt, :, 1 ] )
        UB = fct_ell * ( cl_plt[ i_plt, :, 0 ] + cl_plt[ i_plt, :, 1 ] )
        plt.fill_between( ell, LB, UB, where = UB >= LB, facecolor='blue', alpha= 0.4, zorder = 2 )
        #plt.loglog( ell, LB, '-.b', zorder = 3, alpha = 0.6 )
        #plt.loglog( ell, UB, '-.b', zorder = 3, alpha = 0.6 )
        plt.xlim( [ 2, 3 * Nside ] )
        plt.ylim( [ np.min( fct_ell[ 2 : 3 * Nside - 1 ] * cl_plt[ i_plt, 2 : 3 * Nside - 1, 0 ] ),
                    np.max( fct_ell[ 2 : 3 * Nside - 1 ] * cl_plt[ i_plt, 2 : 3 * Nside - 1, 0 ] ) ] )
        plt.legend( loc = 'lower right', prop = { 'size' : 18 } )
        plt.xlabel( r'$\ell$', fontsize = 18 )
        plt.ylabel( r'$\ell*(\ell+1)*C_{\ell}/(2\pi) \quad (\mu K_{CMB}$)^2', fontsize = 18 )
        plt.grid()
        plt.title( ttl_plt[ i_plt ], fontsize = 18 )
        nm_fig = 'fig/final_summary_%s.%s_nb7_seed100_nside%04i_%s.png' % ( name_in, name_fit, Nside, ttl_plt[ i_plt ] )
        fig.savefig( nm_fig )

    plt.show()


if __name__ == '__main__':
     main()
