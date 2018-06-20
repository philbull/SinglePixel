"""
Utility functions and constants for probabilistic dust models.
"""
import numpy as np
import scipy.integrate as integrate
import scipy.optimize as optimize
import scipy.special as special
import scipy.stats as stats

import model_list
import models

from utils import B_nu, G_nu
from fitting import noise_model

# Physical constants
c = 2.99792458e10       # Speed of light, cm/s
h = 6.62606957e-27      # Planck constant, erg s
k = 1.3806488e-16       # Boltzmann constant, erg/K
Tcmb = 2.7255           # CMB temperature, K

n_samples = 500 # parameter samples
n_freq = 7 # frequency samples
mean_beta = 1.6
mean_temp = 20.
sigma_beta = .2
sigma_temp = 4.
beta_params = [mean_beta, sigma_beta]
temp_params = [mean_temp, sigma_temp]
nu_ref = 353. * 1e9
#nu = np.logspace(np.log10(100), np.log10(800), n_freq) * 1e9
nu = np.logspace(np.log10(30), np.log10(500), n_freq) * 1e9

beta = np.linspace(.1, 5., n_samples)
temp = np.linspace(1., 50., n_samples)
BETA, TEMP = np.meshgrid(beta, temp)

def gaussian(x, params):
    mean, sigma = params # unpack parameters
    normalization = 1. / (np.sqrt(2 * np.pi) * sigma)
    return normalization * np.exp((-1. / ( 2 * sigma**2 )) * (x - mean)**2)

def bimodal(x, params):
    pdf, params1, params2, p = params # unpack parameters
    return p * pdf(x, params1) + (1 - p) * pdf(x, params2)

def prob_MBB(beta, temp, nu, pdf_beta, pdf_temp, beta_params, temp_params, nu_ref = 353. * 1e9):

    joint_pdf = pdf_beta(beta, beta_params) * pdf_temp(temp, temp_params)
    return joint_pdf * single_MBB(nu, (beta, temp, 1.0))

def integrate_I(integrand, nu, beta, temp):
    return [integrate.simps(integrate.simps(integrand(nu[i]), beta), temp) for i in range(len(nu))]

def single_MBB(nu, coeffs):
    # coeffs[0] is beta, coeffs[1] is temp, coeffs[2] is amplitude
    model = coeffs[2] * (nu / nu_ref)**coeffs[0] * B_nu(nu, coeffs[1]) \
               * G_nu(nu_ref, Tcmb) \
               / ( B_nu(nu_ref, coeffs[1]) * G_nu(nu, Tcmb) )
    return model

def residuals(coeffs, y, t):
    return y - np.log10(single_MBB(t, coeffs))

def generate_fits(BETA, TEMP, nu, pdf_beta, pdf_temp, beta_params, temp_params):

    integrand = lambda nu: prob_MBB(BETA, TEMP, nu, pdf_beta, pdf_temp, beta_params,
                                    temp_params)

    integrated_I = integrate_I(integrand, nu, beta, temp)
    I_ref = integrate.simps(integrate.simps(integrand(nu_ref), beta), temp)
    data = (integrated_I, I_ref)

    coeffs_guess = [1.5, 15., 1.05]
    coeffs_fit, flag = optimize.leastsq(residuals, coeffs_guess, args=(np.log10(data[0]), nu))

    return data, coeffs_fit
