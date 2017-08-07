import numpy as np
def bands_log(nu_min, nu_max, nbands):
    """
    Logarithmic set of bands.
    """
    freq_vec = np.arange(nbands)
    return nu_min * (nu_max/nu_min)**(freq_vec/(nbands-1.)) * 1e9 # in Hz

a = np.linspace(30e9, 500e9, 7)
print a
print bands_log(30, 500, 7)
