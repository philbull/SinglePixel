
import numpy as np
from utils import *
from prob_utils import *
from scipy.interpolate import RectBivariateSpline

#-------------------------------------------------------------------------------
# Generalized (2-component) dust models
#-------------------------------------------------------------------------------

class DustGen(object):
    def __init__(self, amp_I, amp_Q, amp_U,
                 beta=1.6, dbeta=0.2, Td1=18., Td2=24., fI=1., fQ=1., fU=1.,
                 name=None):
        """
        Generalized 2-component modified blackbody dust component.
        """
        self.model = 'dustgen'
        self.name = name if name is not None else "DustGen"

        # Reference frequency
        self.nu_ref = 353. * 1e9

        # Conversion factor, 1uK_RJ at 353 GHz to uK_CMB
        nufac = 1. #2.*(353e9)**2. * k / (c**2. * G_nu(353e9, Tcmb))

        # Set amplitude parameters
        self.amp_I = amp_I * nufac
        self.amp_Q = amp_Q * nufac
        self.amp_U = amp_U * nufac

        # Set spectral parameters
        self.beta = beta
        self.dbeta = dbeta
        self.Td1 = Td1
        self.Td2 = Td2
        self.fI = fI
        self.fQ = fQ
        self.fU = fU

        # List of parameter names
        self.param_names = [ 'gdust_beta', 'gdust_dbeta',
                             'gdust_Td1', 'gdust_Td2',
                             'gdust_fI', 'gdust_fQ', 'gdust_fU' ]

    def amps(self):
        """
        Return array of amplitudes, [I, Q, U].
        """
        return np.array([self.amp_I, self.amp_Q, self.amp_U])

    def params(self):
        """
        Return list of parameters.
        """
        return np.array([self.beta, self.dbeta, self.Td1, self.Td2,
                         self.fI, self.fQ, self.fU])

    def set_params(self, params):
        """
        Set parameters from an array, using the same ordering as the list
        returned by self.params().
        """
        self.beta, self.dbeta, self.Td1, self.Td2, \
                           self.fI, self.fQ, self.fU = params

    def scaling(self, nu, params=None):
        """
        Return frequency scaling factor at a given frequency.
        """
        if params is not None:
            beta, dbeta, Td1, Td2, fI, fQ, fU = params
        else:
            beta = self.beta; dbeta = self.dbeta
            Td1 = self.Td1; Td2 = self.Td2
            fI = self.fI; fQ = self.fQ; fU = self.fU
        nu_ref = self.nu_ref

        # Common factor (incl. conversion factor to dT_CMB)
        fac = (nu / nu_ref)**beta * G_nu(nu_ref, Tcmb) / G_nu(nu, Tcmb)

        # Frequency-dependent scalings
        comp1 = B_nu(nu, Td1) / B_nu(nu_ref, Td1)
        comp2 = B_nu(nu, Td2) / B_nu(nu_ref, Td2) * (nu / nu_ref)**dbeta
        gdust_I = fac * (comp1 + fI * comp2)
        gdust_Q = fac * (comp1 + fQ * comp2)
        gdust_U = fac * (comp1 + fU * comp2)

        return np.array([gdust_I, gdust_Q, gdust_U])


class DustGenMBB(DustGen):
    def __init__(self, *args, **kwargs):
        """
        Standard 2-component dust model, with fQ=fU.
        """
        super(DustGenMBB, self).__init__(*args, **kwargs)
        self.model = 'genmbb'
        if self.name is None: self.name = "DustGenMBB"

        # Restrict fU = fQ
        self.fU = self.fQ

        # Reference frequency
        self.nu_ref = 353. * 1e9

        # List of parameter names
        self.param_names = [ 'gdust_beta', 'gdust_dbeta',
                             'gdust_Td1', 'gdust_Td2',
                             'gdust_fI', 'gdust_fQ',]

    def params(self):
        """
        Return list of parameters.
        """
        return np.array([self.beta, self.dbeta, self.Td1, self.Td2,
                         self.fI, self.fQ])

    def set_params(self, params):
        """
        Set parameters from an array, using the same ordering as the list
        returned by self.params().
        """
        self.beta, self.dbeta, self.Td1, self.Td2, self.fI, self.fQ = params
        self.fU = self.fQ

    def scaling(self, nu, params=None):
        """
        Return frequency scaling factor at a given frequency.
        """
        # Make fQ = fU
        if params is not None:
            params = np.concatenate((params, [params[-1],]))
        return super(DustGenMBB, self).scaling(nu, params)


class DustGenMBBDepol(DustGen):
    def __init__(self, *args, **kwargs):
        """
        2-component dust model, with fQ != fU, to allow frequency decorrelation
        effects to be accounted for.
        """
        super(DustGenMBBDepol, self).__init__(*args, **kwargs)
        self.model = 'genmbbdp'
        if self.name is None: self.name = "DustGenMBBDepol"

        # Reference frequency
        self.nu_ref = 353. * 1e9

        # List of parameter names
        self.param_names = [ 'gdust_beta', 'gdust_dbeta',
                             'gdust_Td1', 'gdust_Td2',
                             'gdust_fI', 'gdust_fQ', 'gdust_fU']

    def params(self):
        """
        Return list of parameters.
        """
        return np.array([self.beta, self.dbeta, self.Td1, self.Td2,
                         self.fI, self.fQ, self.fU])

    def set_params(self, params):
        """
        Set parameters from an array, using the same ordering as the list
        returned by self.params().
        """
        self.beta, self.dbeta, self.Td1, self.Td2, \
            self.fI, self.fQ, self.fU = params

    def scaling(self, nu, params=None):
        """
        Return frequency scaling factor at a given frequency.
        """
        return super(DustGenMBBDepol, self).scaling(nu, params)


#-------------------------------------------------------------------------------
# Dust models
#-------------------------------------------------------------------------------

class DustModel(object):
    def __init__(self, amp_I, amp_Q, amp_U,
                 dust_beta=1.6, dust_T=20., fcar=1., fsilfe=0., uval=0.,
                 sigma_beta=None, sigma_temp=None,
                 name=None):
        """
        Generic dust component.
        """
        self.model = 'generic'
        self.name = name

        # Conversion factor, 1uK_RJ at 353 GHz to uK_CMB
        nufac = 1. #2.*(353e9)**2. * k / (c**2. * G_nu(353e9, Tcmb))

        # Set amplitude parameters
        self.amp_I = amp_I * nufac
        self.amp_Q = amp_Q * nufac
        self.amp_U = amp_U * nufac

        # Set spectral parameters
        self.dust_beta = dust_beta
        self.dust_T = dust_T
        self.fcar = fcar
        self.fsilfe = fsilfe
        self.uval = uval
        self.sigma_beta = sigma_beta
        self.sigma_temp = sigma_temp

        # List of parameter names
        self.param_names = ['dust_beta', 'dust_T', 'fcar', 'fsilfe', 'uval']

    def amps(self):
        """
        Return array of amplitudes, [I, Q, U].
        """
        return np.array([self.amp_I, self.amp_Q, self.amp_U])

    def params(self):
        """
        Return list of parameters.
        """
        return np.array([self.dust_beta, self.dust_T, self.fcar,
                         self.fsilfe, self.uval])

    def set_params(self, params):
        """
        Set parameters from an array, using the same ordering as the list
        returned by self.params().
        """
        self.dust_beta, self.dust_T, self.fcar, self.fsilfe, self.uval = params

    def scaling(self, nu, params=None):
        return NotImplementedError("The generic DustModel class does not "
                                   "provide a generic scaling() method.")


class DustMBB(DustModel):
    def __init__(self, *args, **kwargs):
        """
        Modified blackbody dust component.
        """
        #kwargs['hdmodel'] = False
        super(DustMBB, self).__init__(*args, **kwargs)
        self.model = 'mbb'
        if self.name is None: self.name = "DustMBB"

        # Reference frequency
        self.nu_ref = 353. * 1e9

        # List of parameter names
        self.param_names = ['dust_beta', 'dust_T',]

    def params(self):
        """
        Return list of parameters.
        """
        return np.array([self.dust_beta, self.dust_T])

    def set_params(self, params):
        """
        Set parameters from an array, using the same ordering as the list
        returned by self.params().
        """
        self.dust_beta, self.dust_T = params

    def scaling(self, nu, params=None):
        """
        Return frequency scaling factor at a given frequency.
        """
        if params is not None:
            dust_beta, dust_T = params
        else:
            dust_beta = self.dust_beta
            dust_T = self.dust_T
        nu_ref = self.nu_ref

        # Frequency-dependent scalings.
        dust_I = (nu / nu_ref)**dust_beta * B_nu(nu, dust_T) \
               * G_nu(nu_ref, Tcmb) \
               / ( B_nu(353.*1e9, dust_T) * G_nu(nu, Tcmb) )
        dust_Q = dust_I
        dust_U = dust_I

        return np.array([dust_I, dust_Q, dust_U])


class DustSimpleMBB(DustMBB):
    def __init__(self, *args, **kwargs):
        """
        Simplified modified blackbody dust model.
        """
        super(DustSimpleMBB, self).__init__(*args, **kwargs)
        self.model = 'simplembb'
        if self.name is None: self.name = "DustSimpleMBB"

        # List of parameter names
        self.param_names = ['dust_T',]

    def params(self):
        """
        Return list of parameters.
        """
        return np.array([self.dust_T,])

    def set_params(self, params):
        """
        Set parameters from an array, using the same ordering as the list
        returned by self.params().
        """
        self.dust_T = params

    def scaling(self, nu, params=None):
        """
        Return frequency scaling factor at a given frequency.
        """
        if params is not None:
            dust_T = params
            params = np.array([self.dust_beta, dust_T])
        return super(DustSimpleMBB, self).scaling(nu, params)


class DustHD(DustModel):
    def __init__(self, *args, **kwargs):
        """
        Modified blackbody dust component.
        """
        # Initialize the base class
        super(DustHD, self).__init__(*args, **kwargs)

        # Set model name
        self.model = 'hd'
        if self.name is None: self.name = "DustHD"

        # Reference frequency
        self.nu_ref = 353. * 1e9

        # Initialize HD model interpolation functions
        self.initialize_hd_dust_model()

        # List of parameter names
        self.param_names = ['fcar', 'fsilfe', 'uval']

    def initialize_hd_dust_model(self):
        """
        Initialize HD dust model interpolation fns using precomputed data.
        """
        # Read in precomputed dust emission spectra as a fn. of lambda and U
        data_sil = np.genfromtxt("data/sil_fe00_2.0.dat")
        data_silfe = np.genfromtxt("data/sil_fe05_2.0.dat")
        data_car = np.genfromtxt("data/car_1.0.dat")

        wav = data_sil[:,0]
        uvec = np.arange(-3.,5.01,0.1)
        # Units of Jy/sr/H
        sil_i = RectBivariateSpline(uvec,wav,(data_sil[:,3:84] *
                                    (wav[:,np.newaxis]*1.e-4/c)*1.e23).T)
        car_i = RectBivariateSpline(uvec,wav,(data_car[:,3:84] *
                                    (wav[:,np.newaxis]*1.e-4/c)*1.e23).T)
        silfe_i = RectBivariateSpline(uvec,wav,(data_silfe[:,3:84] *
                                    (wav[:,np.newaxis]*1.e-4/c)*1.e23).T)

        sil_p = RectBivariateSpline(uvec,wav,(data_sil[:,84:165] *
                                    (wav[:,np.newaxis]*1.e-4/c)*1.e23).T)
        car_p = RectBivariateSpline(uvec,wav,(data_car[:,84:165] *
                                    (wav[:,np.newaxis]*1.e-4/c)*1.e23).T)
        silfe_p = RectBivariateSpline(uvec,wav,(data_silfe[:,84:165] *
                                    (wav[:,np.newaxis]*1.e-4/c)*1.e23).T)

        # Store inside object
        self.dust_interp = (car_i, sil_i, silfe_i, car_p, sil_p, silfe_p)

    def params(self):
        """
        Return list of parameters.
        """
        return np.array([self.fcar, self.fsilfe, self.uval])

    def set_params(self, params):
        """
        Set parameters from an array, using the same ordering as the list
        returned by self.params().
        """
        self.fcar, self.fsilfe, self.uval = params

    def scaling(self, nu, params=None):
        """
        Return frequency scaling factor at a given frequency.
        """
        if params is not None:
            fcar, fsilfe, uval = params
        else:
            fcar = self.fcar
            fsilfe = self.fsilfe
            uval = self.uval

        nu_ref = self.nu_ref
        car_i, sil_i, silfe_i, car_p, sil_p, silfe_p = self.dust_interp

        # Calculate wavelength and reference wavelength in suitable units
        lam = 1.e4 * c / (nu) # in microns
        lam_ref = 1.e4 * c / nu_ref # in microns

        unit_fac = G_nu(nu_ref, Tcmb) / G_nu(nu, Tcmb)

        # Calculate frequency-dependent scaling factors
        dust_I = unit_fac \
               * (  sil_i.ev(uval, lam)
                  + fcar * car_i.ev(uval, lam)
                  + fsilfe * silfe_i.ev(uval, lam) ) \
               / (  sil_i.ev(uval, lam_ref)
                  + fcar * car_i.ev(uval, lam_ref)
                  + fsilfe * silfe_i.ev(uval, lam_ref) )
        dust_Q = unit_fac \
               * (  sil_p.ev(uval, lam)
                  + fcar * car_p.ev(uval, lam)
                  + fsilfe * silfe_p.ev(uval, lam) ) \
               / (  sil_p.ev(uval, lam_ref)
                  + fcar * car_p.ev(uval, lam_ref)
                  + fsilfe * silfe_p.ev(uval, lam_ref) )
        dust_U = dust_Q

        return np.array([dust_I, dust_Q, dust_U])

#-------------------------------------------------------------------------------
# Probabilistic Dust models
#-------------------------------------------------------------------------------

class ProbSingleMBB(DustModel):
    def __init__(self, *args, **kwargs):
        """
        Modified blackbody dust component.
        """
        #kwargs['hdmodel'] = False

        super(ProbSingleMBB, self).__init__(*args, **kwargs)
        self.model = 'prob1mbb'
        if self.name is None: self.name = "Prob1MBB"

        # Reference frequency
        self.nu_ref = 353. * 1e9
        self.pdf_beta = gaussian
        self.pdf_temp = gaussian
        # List of parameter names
        self.param_names = ['dust_beta', 'dust_T', 'sigma_beta', 'sigma_temp'] # explicit names ex. mean_beta

    def params(self):
        """
        Return list of parameters.
        """
        return np.array([self.dust_beta, self.dust_T, self.sigma_beta, self.sigma_temp])

    def set_params(self, params):
        """
        Set parameters from an array, using the same ordering as the list
        returned by self.params().
        """
        # not correct for probabilistic model, FIX ME
        self.dust_beta, self.dust_T, self.sigma_beta, self.sigma_temp = params

    def scaling(self, nu, params=None):
        """
        Return frequency scaling factor at a given frequency.
        """
        if params is not None:
            dust_beta, dust_T, sigma_beta, sigma_temp = params
        else:
            dust_beta = self.dust_beta
            dust_T = self.dust_T
            sigma_beta = self.sigma_beta
            sigma_temp = self.sigma_temp

        nu_ref = self.nu_ref

        n_samples = 500
        beta = np.linspace(.1, 5., n_samples)
        temp = np.linspace(1., 50., n_samples)
        BETA, TEMP = np.meshgrid(beta, temp)

        integrand = lambda nu: prob_MBB(BETA, TEMP, nu, gaussian, gaussian, (dust_beta, sigma_beta),
         (dust_T, sigma_temp))

        I_ref = integrate.simps(integrate.simps(integrand(nu_ref), beta), temp)

        # Frequency-dependent scalings.
        dust_I = [integrate.simps(integrate.simps(integrand(nu), beta), temp) / I_ref for nu in nu]
        dust_Q = dust_I
        dust_U = dust_I

        return np.array([dust_I, dust_Q, dust_U])

# UPDATE ME TO MATCH NEW ProbSingleMBB
class ProbTwoMBB(DustGen):
    def __init__(self, *args, **kwargs):
        """
        Modified blackbody dust component.
        """
        #kwargs['hdmodel'] = False

        super(ProbSingleMBB, self).__init__(*args, **kwargs)
        self.model = 'prob1mbb'
        if self.name is None: self.name = "Prob1MBB"

        # Reference frequency
        self.nu_ref = 353. * 1e9
        self.pdf_beta = gaussian
        self.pdf_temp = gaussian
        # List of parameter names
        self.param_names = ['dust_beta', 'dust_T', 'sigma_beta', 'sigma_temp'] # explicit names ex. mean_beta

    def params(self):
        """
        Return list of parameters.
        """
        return np.array([self.dust_beta, self.dust_T, self.sigma_beta, self.sigma_temp])

    def set_params(self, params):
        """
        Set parameters from an array, using the same ordering as the list
        returned by self.params().
        """
        # not correct for probabilistic model, FIX ME
        self.dust_beta, self.dust_T, self.sigma_beta, self.sigma_temp = params

    def scaling(self, nu, params=None):
        """
        Return frequency scaling factor at a given frequency.
        """
        if params is not None:
            dust_beta, dust_T, sigma_beta, sigma_temp = params
        else:
            dust_beta = self.dust_beta
            dust_T = self.dust_T
            sigma_beta = self.sigma_beta
            sigma_temp = self.sigma_temp

        nu_ref = self.nu_ref

        n_samples = 500
        beta = np.linspace(.1, 5., n_samples)
        temp = np.linspace(1., 50., n_samples)
        BETA, TEMP = np.meshgrid(beta, temp)

        integrand = lambda nu: prob_MBB(BETA, TEMP, nu, gaussian, gaussian, (dust_beta, sigma_beta),
         (dust_T, sigma_temp))

        I_ref = integrate.simps(integrate.simps(integrand(nu_ref), beta), temp)

        # Frequency-dependent scalings.
        dust_I = [integrate.simps(integrate.simps(integrand(nu), beta), temp) / I_ref for nu in nu]
        dust_Q = dust_I
        dust_U = dust_I

        return np.array([dust_I, dust_Q, dust_U])

#-------------------------------------------------------------------------------
# AME model
#-------------------------------------------------------------------------------

class AMEModel(object):
    def __init__(self, amp_I, amp_Q, amp_U, nu_peak, name=None):
        """
        Simple AME component.
        """
        self.model = 'ame'
        self.name = "AME" if name is None else name

        # Reference frequency
        self.nu_ref = 30.*1.e9 # Hz

        # Conversion factor, 1uK_RJ at 30 GHz to uK_CMB
        #nufac = 2.*(self.nu_ref)**2. * k \
        #      / (c**2. * G_nu(self.nu_ref, Tcmb))
        nufac = 1.

        # Set amplitude parameters
        self.amp_I = amp_I * nufac
        self.amp_Q = amp_Q * nufac
        self.amp_U = amp_U * nufac

        # Set spectral parameters
        self.nu_peak = nu_peak

        # List of parameter names
        self.param_names = ['ame_nupeak',]

    def amps(self):
        """
        Return array of amplitudes, [I, Q, U].
        """
        return np.array([self.amp_I, self.amp_Q, self.amp_U])

    def params(self):
        """
        Return list of parameters.
        """
        return np.array([self.nu_peak,])

    def set_params(self, params):
        """
        Set parameters from an array, using the same ordering as the list
        returned by self.params().
        """
        self.nu_peak = params

    def scaling(self, nu, params=None):
        """
        Return frequency scaling factor at a given frequency.
        """
        # Peak freq.
        if params is not None:
            nu_peak = params * 1e9 # in Hz
        else:
            nu_peak = self.nu_peak * 1e9 # in Hz

        # Reference amplitude
        ref = (self.nu_ref / nu_peak)**2. \
            * np.exp(1. - (self.nu_ref / nu_peak)**2.) \
            * G_nu(nu, Tcmb) / G_nu(self.nu_ref, Tcmb)

        # Frequency scalings
        ame_I = (nu / nu_peak)**2. * np.exp(1. - (nu / nu_peak)**2.) / ref
        ame_Q = ame_I
        ame_U = ame_I

        return np.array([ame_I, ame_Q, ame_U])


#-------------------------------------------------------------------------------
# Synchrotron model
#-------------------------------------------------------------------------------

class SyncModel(object):
    def __init__(self, amp_I, amp_Q, amp_U, sync_beta, name=None):
        """
        Generic synchrotron component.
        """
        self.model = 'generic'
        self.name = "Sync" if name is None else name

        # Reference frequency
        self.nu_ref = 30.*1.e9 # Hz

        # Conversion factor, 1uK_RJ at 30 GHz to uK_CMB
        #nufac = 2.*(self.nu_ref)**2. * k \
        #      / (c**2. * G_nu(self.nu_ref, Tcmb))
        nufac = 1.

        # Set amplitude parameters
        self.amp_I = amp_I * nufac
        self.amp_Q = amp_Q * nufac
        self.amp_U = amp_U * nufac

        # Set spectral parameters
        self.sync_beta = sync_beta

        # List of parameter names
        self.param_names = ['sync_beta',]

    def amps(self):
        """
        Return array of amplitudes, [I, Q, U].
        """
        return np.array([self.amp_I, self.amp_Q, self.amp_U])

    def params(self):
        """
        Return list of parameters.
        """
        return np.array([self.sync_beta,])

    def set_params(self, params):
        """
        Set parameters from an array, using the same ordering as the list
        returned by self.params().
        """
        self.sync_beta = params

    def scaling(self, nu, params=None):
        """
        Return frequency scaling factor at a given frequency.
        """
        if params is not None:
            sync_beta = params
        else:
            sync_beta = self.sync_beta

        # Frequency scaling
        sync_I = (nu / self.nu_ref)**sync_beta \
               * G_nu(self.nu_ref, Tcmb) / G_nu(nu, Tcmb)
        sync_Q = sync_I
        sync_U = sync_I

        return np.array([sync_I, sync_Q, sync_U])


class SyncPow(SyncModel):
    def __init__(self, *args, **kwargs):
        """
        Powerlaw synchrotron component.
        """
        super(SyncPow, self).__init__(*args, **kwargs)
        self.model = 'pow'
        if self.name is None: self.name = "SyncPowerlaw"


#-------------------------------------------------------------------------------
# Free-free model
#-------------------------------------------------------------------------------

class FreeFreeModel(object):
    def __init__(self, amp_I, amp_Q, amp_U, ff_beta=-0.118, name=None):
        """
        Generic free-free component.
        """
        self.model = 'generic'
        self.name = name

        self.nu_ref = 30. * 1e9 # Reference frequency

        # Conversion factor, 1uK_RJ at 30 GHz to uK_CMB
        #nufac = 2.*(self.nu_ref)**2. * k \
        #      / (c**2. * G_nu(self.nu_ref, Tcmb))
        nufac = 1.

        # Set amplitude parameters
        self.amp_I = amp_I * nufac
        self.amp_Q = amp_Q * nufac
        self.amp_U = amp_U * nufac

        # Set spectral parameters
        self.ff_beta = ff_beta

        # List of parameter names
        self.param_names = ['ff_beta',]

    def amps(self):
        """
        Return array of amplitudes, [I, Q, U].
        """
        return np.array([self.amp_I, self.amp_Q, self.amp_U])

    def params(self):
        """
        Return list of parameters.
        """
        return np.array([self.ff_beta,])

    def set_params(self, params):
        """
        Set parameters from an array, using the same ordering as the list
        returned by self.params().
        """
        self.ff_beta = params

    def scaling(self, nu, params=None):
        """
        Return frequency scaling factor at a given frequency.
        """
        if params is not None:
            ff_beta = params
        else:
            ff_beta = self.ff_beta

        # Frequency scaling
        ff_I = (nu / self.nu_ref)**ff_beta \
             * G_nu(self.nu_ref, Tcmb) / G_nu(nu, Tcmb)
        ff_Q = ff_I
        ff_U = ff_I
        return np.array([ff_I, ff_Q, ff_U])


class FreeFreePow(FreeFreeModel):
    def __init__(self, *args, **kwargs):
        """
        Powerlaw free-free component.
        """
        super(FreeFreePow, self).__init__(*args, **kwargs)
        self.model = 'pow'
        if self.name is None: self.name = "FFPowerlaw"


class FreeFreeUnpol(FreeFreeModel):
    def __init__(self, *args, **kwargs):
        """
        Powerlaw free-free component with no polarisation.
        """
        super(FreeFreeUnpol, self).__init__(*args, **kwargs)
        self.model = 'ffunpol'
        if self.name is None: self.name = "FFUnpol"

        self.param_names = []

    def params(self):
        """
        Return list of parameters.
        """
        return np.array([])

    def set_params(self, params):
        """
        Set parameters from an array, using the same ordering as the list
        returned by self.params().
        """
        # No parameters; do nothing

    def scaling(self, nu, params=None):
        """
        Return frequency scaling factor at a given frequency.
        """
        ff_I = (nu / self.nu_ref)**self.ff_beta \
             * G_nu(self.nu_ref, Tcmb) / G_nu(nu, Tcmb)
        ff_Q = 0. * ff_I
        ff_U = 0. * ff_I
        return np.array([ff_I, ff_Q, ff_U])

#-------------------------------------------------------------------------------
# CMB model
#-------------------------------------------------------------------------------

class CMB(object):
    def __init__(self, amp_I, amp_Q, amp_U, name=None):
        """
        CMB component.
        """
        self.model = 'cmb'
        self.name = "CMB" if name is None else name

        # Set amplitude parameters
        self.amp_I = amp_I
        self.amp_Q = amp_Q
        self.amp_U = amp_U

        # List of parameter names
        self.param_names = []

    def amps(self):
        """
        Return array of amplitudes, [I, Q, U].
        """
        return np.array([self.amp_I, self.amp_Q, self.amp_U])

    def params(self):
        """
        Return list of parameters.
        """
        return np.array([])

    def set_params(self, params):
        """
        Set parameters from an array, using the same ordering as the list
        returned by self.params().
        """
        pass

    def scaling(self, nu, params=None):
        """
        Return frequency scaling factor at a given frequency.
        """
        cmb_I = np.ones(len(nu))
        cmb_Q = cmb_I
        cmb_U = cmb_I
        return np.array([cmb_I, cmb_Q, cmb_U])
