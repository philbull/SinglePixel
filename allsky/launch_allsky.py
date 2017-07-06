# Simple program to launch a serial job for the all sky fitting
# mpiexec -n 4 python launch_allsky.py
import os
import numpy as np
import healpy as hp
from mpi4py import MPI

import model_values_allsky
import model_list_allsky
import run_joint_mcmc_allsky_split

import time

def main( in_list = 'sync,freefree,mbb', 
          fit_list = 'sync,freefree,mbb',
          Nside = 2,
          nu_min = 15,
          nu_max = 800,
          n_bands = 7,
          seed = 100,
          redo = 0 ):

    # 1) Set-up MPI
    comm = MPI.COMM_WORLD
    my_id = comm.Get_rank()
    n_proc = comm.Get_size()
    t0_run = time.time()

    # 2) Preparing the list of with the input and fit model
    in_list_all = [ 'cmb', ] ; fit_list_all = [ 'cmb', ]
    # Removing any white spaces
    in_list = "".join( in_list.split( ) )
    # Creating a list
    in_list_all += in_list.split( "," )
    fit_list = "".join( fit_list.split( ) )
    fit_list_all += fit_list.split( "," )
    # Getting the file name and other useful labels
    amp_names_in, param_names_in, amp_names_fit, param_names_fit, name_in, name_fit = \
    model_list_allsky.get_model_param_names( in_list = in_list, fit_list = fit_list )
    # Getting the number of variables that will be stored
    N_fit = len( amp_names_fit ) + len( param_names_fit )
    # output directory
    out_dir = "output"
    if ( os.path.isdir( out_dir ) != 1 ):
        os.system( 'mkdir %s' % out_dir )
    # Name for the results
    filename = '%s/allsky_summary_%s.%s_nb7_seed%04i_nside%04i.fits' % ( out_dir, name_in, name_fit, seed, Nside )
 
    if my_id == 0:
        print( "(LAUNCH_ALLSKY) Running in a system with %i processors" % ( n_proc ) )
        # Defaut list of allowed models
        allowed_comps = model_list_allsky.model_dict()
        # Make sure models are of known types
        for item in in_list_all:
            if item not in allowed_comps.keys():
                raise ValueError("Unknown component type '%s'" % item)

        for item in fit_list_all:
            if item not in allowed_comps.keys():
                raise ValueError("Unknown component type '%s'" % item)

        # 2.2) Checking the work done
        # If it is not to be redone and the file exists, open the file
        if ( redo == 0 ) and ( os.path.isfile( filename ) == 1 ):
            maps = hp.read_map( filename, 0 )
            # Get pixels that have not been run
	    px_unseen = np.where( maps == hp.UNSEEN )
            # Tuple to array
            px_unseen = px_unseen[ 0 ]
        # If we want to re-do the analysis, remove the file
        if ( redo == 1 ) and ( os.path.isfile( filename ) == 1 ):
            os.system( 'rm %s' % ( filename ) )
        # If it does no exist, create the file filled with UNSEEN values
        if ( os.path.isfile( filename ) != 1 ):
            px_unseen = np.arange( 12 * Nside * Nside )
            maps = np.full( ( 3 * N_fit, 12 * Nside * Nside ), hp.UNSEEN, dtype = 'float32' )
            hp.write_map( filename, maps, dtype = 'float32' )

        # 2.3) If there is no work to do
        n_unseen = len( px_unseen )
        # Else, run it
        if n_unseen != 0:
            print "(LAUNCH_ALLSKY) There are still %i pixels to be analyzed. Running ...'" % ( n_unseen )
            # Create array of dictionaries from the global dictionary, which is created for the missing pixels to be run 
            allsky_dict_splt = model_values_allsky.get_allsky_dict_split( in_list, Nside, px_unseen, seed, n_proc )
            # Create array to store the results
            rslts_all = np.array_split( np.full( ( n_unseen, 3 * N_fit ), hp.UNSEEN, dtype = 'float32' ), n_proc )
            px_all = np.array_split( px_unseen, n_proc )
    else:
        allsky_dict_splt = None
        n_unseen = None
        rslts_all = None
        px_all = None

    # 3) MPI I/O
    # 3.1) Broadcasting whether there is any work to do
    n_unseen = comm.bcast( n_unseen, root = 0 )
    if n_unseen == 0:
        print "(LAUNCH_ALLSKY) No more pixels to analyze in proc %i. Results are found in: %s" % ( my_id, filename )
        return

    # 3.2) The portion of the allsky dictionary that gets assigned to each processor
    dict_splt = comm.scatter( allsky_dict_splt, root = 0 )
    # 3.3) The portion of the results that will be considered
    rslts_splt = comm.scatter( rslts_all, root = 0 )
    # 3.4) The pixel numbers that will be considered (for intermediate storage only)
    px_splt = comm.scatter( px_all, root = 0 )
    
    # 4) MPI runs
    # Number of pixels to consider in a processor
    n_px = len( dict_splt['cmb'][ 0 ] )
    # Timing
    t0 = time.time() 
    t1 = 0
    for i_px in range( n_px ):
        # 4.1) Run single pixel with dict_splt
        rslts_splt[ i_px ] = run_joint_mcmc_allsky_split.main( 
    			in_list = in_list_all,
			fit_list = fit_list_all,
			nu_min = nu_min,
			nu_max = nu_max,
               	        n_bands = n_bands,
			input_dict = dict_splt,
  	              	idx_px = i_px,
        	        seed = i_px + 2 * n_px * my_id,  # Distinct seed. Notice that n_px may vary slightly from process to process -> '2'
			Nside = Nside )
        t1 = t1 + time.time() - t0
        t0 = time.time()
        if ( np.ceil( ( i_px + 1 ) / 10. ) == ( ( i_px + 1 ) / 10. ) ):
            print( "(LAUNCH_ALLSKY) %2.1f %s of the run in processor %i done. ~%2.1f minutes for completion. ETA:  %s" % ( ( 100. * i_px / n_px ), '%', my_id, 1. * ( n_px - i_px ) * ( t1 ) / i_px / 60., time.ctime(time.time() + 1. * ( n_px - i_px ) * ( t1 ) / i_px ) ) )

        # Option of storing partial results (no need to gather since this is a temporary result)
        if ( np.ceil( ( i_px + 1 ) / 20. ) == ( ( i_px + 1 ) / 20. ) ) and ( i_px != n_px - 1 ):
            try:
                maps_tmp = hp.read_map( filename, np.arange( 3 * N_fit ) )
                rslts_tmp = np.transpose( rslts_splt )
                for ii in range( 3 * N_fit ):
                    maps_tmp[ ii ][ px_splt[ 0 : i_px ] ] = rslts_tmp[ ii ][ 0 : i_px ]
                hp.write_map( filename, maps_tmp, dtype = 'float32' )
                print( "(LAUNCH_ALLSKY) Intermediate data written (My_ID=%i)" % my_id )
            except ( IOError, ValueError, UnboundLocalError ): # These are probably all, but add other excpetion types if necessary.
                print( "(LAUNCH_ALLSKY) Could not write intermediate data (My_ID=%i). Continuing ..." % my_id )
                pass

    # 4.2) Gathering the results
    comm.Barrier()
    rslts_all = comm.gather( rslts_splt, root = 0 )
    # 4.3) Writing the results
    if my_id == 0: 
        # Joining the different pieces into the two-dimensional structure to be stored
        data_fits = np.concatenate( rslts_all )
        # Re-shaping the array to be stored (alternatively, one could store it as n_px * (3*N_fit) but it looks less intuitive for reading and plotting)
        maps = hp.read_map( filename, range( 3 * N_fit ) )
        data_fits_tmp = np.transpose( data_fits )
        for ii in range( 3 * N_fit ):
            maps[ ii ][ px_unseen ] = data_fits_tmp[ ii ][ : ]
        # NB: Letting try/exception inherited from pyfits (embedded in healpy) take care of any eventual issue (it should never happen indeed unless major IO issue)
        hp.write_map( filename, maps, dtype = 'float32' )
        print( 'Results written in %s' % filename )
        t_run = ( time.time() - t0_run ) / 3600
        str_run = 'hours'
        if t_run < 1:
            t_run *= 60
            str_run = 'minutes'
        print( 'Total running time %2.2f %s' % ( t_run, str_run ) )

if __name__ == '__main__':
     main()
