"""
Utility functions and constants.
"""
import numpy as np

# Physical constants
c = 2.99792458e10       # Speed of light, cm/s
h = 6.62606957e-27      # Planck constant, erg s
k = 1.3806488e-16       # Boltzmann constant, erg/K
Tcmb = 2.7255           # CMB temperature, K

def B_nu(nu, T):
    """
    Planck blackbody function.
    """
    return 2. * h * nu**3. / (c**2. * np.expm1(h * nu / (k*T)) )


def G_nu(nu, T):
    """
    Conversion factor from intensity to T_CMB, i.e. I_nu = G_nu * deltaT_CMB.
    """
    x = h * nu / (k * T)
    return B_nu(nu, T) * x * np.exp(x) / (np.expm1(x) * T)

def rj2cmb(nu, T):
    """
    Convert a Rayleigh-Jeans temperature to a CMB temperature.
    """
    return 2. * k * (nu / c)**2. * T / G_nu(nu, Tcmb)

def cmb2rj(nu, dT):
    """
    Convert a CMB temperature to a Rayleigh-Jeans temperature.
    """
    return dT * G_nu(nu, Tcmb) / ( 2. * k * (nu / c)**2. )

def bands_log(nu_min, nu_max, nbands):
    """
    Logarithmic set of bands.
    """
    freq_vec = np.arange(nbands)
    return nu_min * (nu_max/nu_min)**(freq_vec/(nbands-1.)) * 1e9 # in Hz
