# Simple program to launch a serial job for the all sky fitting
import os
import healpy as hp

import pdb

def main( in_list = 'sync,mbb', 
          fit_list = 'sync,mbb',
          Nside = 32,
          redo = 0 ):

    in_list_all = [ 'cmb', ] ; fit_list_all = [ 'cmb', ]
    # Removing any white spaces
    in_list = "".join( in_list.split( ) )
    # Creating a list
    in_list_all += in_list.split( "," )
    fit_list = "".join( fit_list.split( ) )
    fit_list_all += fit_list.split( "," )
    # Getting the file name
    name_in = "-".join( in_list_all )
    name_fit = "-".join( fit_list_all )
    # Default
    q_unseen = [ True ] * ( 12 * Nside * Nside )
    # If it is not to be redone and the file exists, open the file
    filename = 'output/final_summary_%s.%s_nb7_seed100_nside%04i.fits' % ( name_in, name_fit, Nside )
    if ( redo == 0 ) and ( os.path.isfile( filename ) == 1 ):
	maps = hp.read_map( filename, 0 )
	q_unseen = ( maps == hp.UNSEEN )
    if ( redo == 1 ) and ( os.path.isfile( filename ) == 1 ):
        os.system( 'rm %s' % ( filename ) )
    # Loop
    for Npix in range( 12 * Nside * Nside ):
        if q_unseen[ Npix ]:
	    cmmnd = 'python run_joint_mcmc_allsky.py 100 "%s" "%s" %i %i' % ( in_list, fit_list, Nside, Npix )
	    print cmmnd
	    os.system( cmmnd )

if __name__ == '__main__':
     main()
