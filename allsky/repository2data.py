# Function to prepare the data for the all-sky analysis
# The defaut convention in healpy is *Ring* ordering

import healpy as hp
import numpy as np
# Plotting tests
import matplotlib.pyplot as plt

import pdb

def main( Nside = 32, plot = 0 ):
    
    # Synchrotron: ***data are in micro K_RJ***
    ## Opening the data from Planck (Commander)
    ### Intensity is given at the reference frequency of 408 MHz
    sync_I = hp.read_map( "repository/COM_CompMap_Synchrotron-commander_0256_R2.00.fits", hdu = 1, field = 0 )
    ### Polarization is given at the reference frequency of 30 GHz
    sync_Q = hp.read_map( "repository/COM_CompMap_SynchrotronPol-commander_0256_R2.00.fits", hdu = 1, field  = 0 )
    sync_U = hp.read_map( "repository/COM_CompMap_SynchrotronPol-commander_0256_R2.00.fits", hdu = 1, field = 1 )
    ## Map of spectral index from WMAP (Lambda site, WMAP 9 year products, model f)
    beta_sync = hp.read_map( "repository/wmap_mcmc_fs_synch_spec_index_9yr_v5.fits" )
    ## Adapting it to the final nside
    beta_sync_nside = hp.ud_grade( beta_sync, nside )
    sync_out = hp.ud_grade( [ sync_I, sync_Q, sync_U ], nside )
    ## Conversion factor from 408 MHz to 30 GHz
    sync_out[ 0 ] = sync_out[ 0 ] * ( 30. / 0.408 )**beta_sync_nside
    # Storing the results
    filename = "data/sync_IQU_%04d.fits" % ( nside )
    hp.write_map( filename, sync_out )
    filename_beta = "data/sync_beta_%04d.fits" % ( nside )
    hp.write_map( filename_beta, beta_sync_nside )


    if ( plot ):
        sync_mp = hp.read_map( filename, np.arange( 3 ) )
        ttl_stks = [ 'I', 'Q', 'U' ]
        for i_stks in range( 3 ):
            hp.mollview( sync_mp[i_stks], title="Synchrotron map %s (nside=%d, pix=%2.1f arcmin)" % ( ttl_stks[ i_stks ], nside, 3600. / nside ), 
                         norm='hist', unit='micro K_RJ @ 30 GHz' ) #, sub = ( 3, i_stks + 1, 1 ) ) 
        plt.show( block = False )

    # Dust: ***data are given in micro K_RJ***
    ## Opening the data from Planck (Commander)
    ### Temperature is given at the reference frequency of 545 GHz
    dust_I = hp.read_map( "repository/COM_CompMap_dust-commander_0256_R2.00.fits", hdu = 1, field = 0 ) 
    ### Polarization is given at the reference frequency of 353 GHz
    dust_Q = hp.read_map( "repository/COM_CompMap_DustPol-commander_1024_R2.00.fits", hdu = 1, field = 0 ) 
    dust_U = hp.read_map( "repository/COM_CompMap_DustPol-commander_1024_R2.00.fits", hdu = 1, field = 1 )
    ## Dust temperature and emissivity
    temp_dust = hp.read_map( "repository/COM_CompMap_dust-commander_0256_R2.00.fits", hdu = 1, field = 3 )
    beta_dust = hp.read_map( "repository/COM_CompMap_dust-commander_0256_R2.00.fits", hdu = 1, field = 6 )
    ## Adapting them to the final nside
    dust_I_nside = hp.ud_grade( dust_I, nside )
    dust_Q_nside = hp.ud_grade( dust_Q, nside )
    dust_U_nside = hp.ud_grade( dust_U, nside )
    temp_dust_nside = hp.ud_grade( temp_dust, nside )
    beta_dust_nside = hp.ud_grade( beta_dust, nside )
    ## Conversion factor from 545 GHz to 353 GHz
    ### Physical constants (SI)
    const_h = 6.6260755e-34
    const_k = 1.3806580e-23
    gamma = const_h / ( const_k * temp_dust_nside )
    dust_I_nside = dust_I_nside * ( 353./ 545. )**( beta_dust_nside + 1. ) * np.expm1( gamma * 545e9 ) / np.expm1( gamma * 353e9 )
    dust_out = [ dust_I_nside, dust_Q_nside, dust_U_nside ]
    ## Storing the results
    filename = "data/dust_IQU_%04d.fits" % ( nside )
    hp.write_map( filename, dust_out )
    filename_temp = "data/dust_temp_%04d.fits" % ( nside )
    hp.write_map( filename_temp, temp_dust_nside )
    filename_beta = "data/dust_beta_%04d.fits" % ( nside )
    hp.write_map( filename_beta, beta_dust_nside )

    if ( plot ):
        dust_mp = hp.read_map( filename, np.arange( 3 ) )
        ttl_stks = [ 'I', 'Q', 'U' ]
        for i_stks in range( 3 ):
            hp.mollview( dust_mp[i_stks], title="Dust map %s (nside=%d, pix=%2.1f arcmin)" % ( ttl_stks[ i_stks ], nside, 3600. / nside ),
                         norm='hist', unit='micro K_RJ @ 353 GHz' ) #, sub = ( 3, i_stks + 1, 1 ) )
        dust_temp = hp.read_map( filename_temp )
        hp.mollview( dust_temp, title="Dust temperature %s (nside=%d, pix=%2.1f arcmin)" % ( ttl_stks[ i_stks ], nside, 3600. / nside ),
                         norm='hist', unit='K' )
        dust_beta = hp.read_map( filename_beta )
        hp.mollview( dust_beta, title="Dust beta %s (nside=%d, pix=%2.1f arcmin)" % ( ttl_stks[ i_stks ], nside, 3600. / nside ),
                         norm='hist', unit='Adimensional' ) 
        plt.show( block = False )


if __name__ == '__main__':
     main()
