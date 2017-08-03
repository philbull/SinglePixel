# Function to prepare the data for the all-sky analysis
# The defaut convention in healpy is *Ring* ordering

import healpy as hp
import numpy as np
# Plotting tests
import matplotlib.pyplot as plt

import pdb

def main( Nside = 32, plot = 0 ):
    
    # Synchrotron: ***public data are in micro K_RJ***
    ## Opening the data from Planck (Commander)
    ### Intensity is given at the reference frequency of 408 MHz
    sync_I = hp.read_map( "repository/COM_CompMap_Synchrotron-commander_0256_R2.00.fits", hdu = 1, field = 0 )
    ### Polarization is given at the reference frequency of 30 GHz
    sync_Q = hp.read_map( "repository/COM_CompMap_SynchrotronPol-commander_0256_R2.00.fits", hdu = 1, field  = 0 )
    sync_U = hp.read_map( "repository/COM_CompMap_SynchrotronPol-commander_0256_R2.00.fits", hdu = 1, field = 1 )
    ## Map of spectral index from WMAP (Lambda site, WMAP 9 year products, model f)
    beta_sync = hp.read_map( "repository/wmap_mcmc_fs_synch_spec_index_9yr_v5.fits" )
    ## Adapting it to the final Nside
    beta_sync_Nside = hp.ud_grade( beta_sync, Nside )
    sync_out = hp.ud_grade( [ sync_I, sync_Q, sync_U ], Nside )
    ## Conversion factor from 408 MHz to 30 GHz (reference frequency in the all sky code)
    sync_out[ 0 ] = sync_out[ 0 ] * ( 30. / 0.408 )**beta_sync_Nside
    # Storing the results
    filename = "data/sync_IQU_%04d.fits" % ( Nside )
    hp.write_map( filename, sync_out )
    filename_beta = "data/sync_beta_%04d.fits" % ( Nside )
    hp.write_map( filename_beta, beta_sync_Nside )


    if ( plot ):
        sync_mp = hp.read_map( filename, np.arange( 3 ) )
        ttl_stks = [ 'I', 'Q', 'U' ]
        for i_stks in range( 3 ):
            hp.mollview( sync_mp[i_stks], title="Synchrotron map %s (Nside=%d, pix=%2.1f arcmin)" % ( ttl_stks[ i_stks ], Nside, 3600. / Nside ), 
                         norm='hist', unit='micro K_RJ @ 30 GHz' ) #, sub = ( 3, i_stks + 1, 1 ) ) 
        plt.show( block = False )

    # FreeFree. For now, from https://lambda.gsfc.nasa.gov/product/map/dr5/mcmc_maps_fs_get.cfm. Commander free-free has an odd structure in temperature.
    ## Units say K, but since K_RJ ~ K_CMB at 22 GHz, I am simply assuming K_RJ.
    ### Intensity is given at the reference frequency of 22 GHz (K band)
    ff_I = hp.read_map( "repository/wmap_mcmc_fs_k_freefree_temp_9yr_v5.fits", hdu = 1, field = 0 )
    ### Polarization is zero (Obviously could be removed as a function argument, but I keep it as in the single pixel coding case)
    ff_Q = 0
    ff_U = 0
    ## spectral index coNside flat for now
    beta_ff = -0.118
    ## Adapting it to the final Nside
    ff_out = hp.ud_grade( ff_I, Nside )
    ## Conversion factor from 22 GHz to 30 GHz (reference frequency in the all sky code)
    ff_out[ 0 ] = ff_out[ 0 ] * ( 30. / 22 )**beta_ff
    # Storing the results (only I)
    filename = "data/freefree_I_%04d.fits" % ( Nside )
    hp.write_map( filename, ff_out )


    # Dust: ***public data are given in micro K_RJ***
    ## Opening the data from Planck (Commander)
    ### Temperature is given at the reference frequency of 545 GHz
    dust_I = hp.read_map( "repository/COM_CompMap_dust-commander_0256_R2.00.fits", hdu = 1, field = 0 ) 
    ### Polarization is given at the reference frequency of 353 GHz
    dust_Q = hp.read_map( "repository/COM_CompMap_DustPol-commander_1024_R2.00.fits", hdu = 1, field = 0 ) 
    dust_U = hp.read_map( "repository/COM_CompMap_DustPol-commander_1024_R2.00.fits", hdu = 1, field = 1 )
    ## Dust temperature and emissivity
    temp_dust = hp.read_map( "repository/COM_CompMap_dust-commander_0256_R2.00.fits", hdu = 1, field = 3 )
    beta_dust = hp.read_map( "repository/COM_CompMap_dust-commander_0256_R2.00.fits", hdu = 1, field = 6 )
    ## Adapting them to the final Nside
    dust_I_Nside = hp.ud_grade( dust_I, Nside )
    dust_Q_Nside = hp.ud_grade( dust_Q, Nside )
    dust_U_Nside = hp.ud_grade( dust_U, Nside )
    temp_dust_Nside = hp.ud_grade( temp_dust, Nside )
    beta_dust_Nside = hp.ud_grade( beta_dust, Nside )
    ## Conversion factor from 545 GHz to 353 GHz (reference frequency in the all sky code)
    ### Physical constants (SI)
    const_h = 6.6260755e-34
    const_k = 1.3806580e-23
    gamma = const_h / ( const_k * temp_dust_Nside )
    dust_I_Nside = dust_I_Nside * ( 353./ 545. )**( beta_dust_Nside + 1. ) * np.expm1( gamma * 545e9 ) / np.expm1( gamma * 353e9 )
    dust_out = [ dust_I_Nside, dust_Q_Nside, dust_U_Nside ]
    ## Storing the results
    filename = "data/dust_IQU_%04d.fits" % ( Nside )
    hp.write_map( filename, dust_out )
    filename_temp = "data/dust_temp_%04d.fits" % ( Nside )
    hp.write_map( filename_temp, temp_dust_Nside )
    filename_beta = "data/dust_beta_%04d.fits" % ( Nside )
    hp.write_map( filename_beta, beta_dust_Nside )

    if ( plot ):
        dust_mp = hp.read_map( filename, np.arange( 3 ) )
        ttl_stks = [ 'I', 'Q', 'U' ]
        for i_stks in range( 3 ):
            hp.mollview( dust_mp[i_stks], title="Dust map %s (Nside=%d, pix=%2.1f arcmin)" % ( ttl_stks[ i_stks ], Nside, 3600. / Nside ),
                         norm='hist', unit='micro K_RJ @ 353 GHz' ) #, sub = ( 3, i_stks + 1, 1 ) )
        dust_temp = hp.read_map( filename_temp )
        hp.mollview( dust_temp, title="Dust temperature %s (Nside=%d, pix=%2.1f arcmin)" % ( ttl_stks[ i_stks ], Nside, 3600. / Nside ),
                         norm='hist', unit='K' )
        dust_beta = hp.read_map( filename_beta )
        hp.mollview( dust_beta, title="Dust beta %s (Nside=%d, pix=%2.1f arcmin)" % ( ttl_stks[ i_stks ], Nside, 3600. / Nside ),
                         norm='hist', unit='Adimensional' ) 
        plt.show( block = False )


if __name__ == '__main__':
     main()
