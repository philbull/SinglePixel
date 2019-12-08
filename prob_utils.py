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

chi = np.linspace(-np.pi / 2., np.pi / 2., 500)
mean_chi = np.pi / 8.
kappa = 1.
X = 4
mean_T = mean_temp

def gaussian(x, params):
    mean, sigma = params # unpack parameters
    normalization = 1. / (np.sqrt(2 * np.pi) * sigma)
    return normalization * np.exp((-1. / ( 2 * sigma**2. )) * (x - mean)**2.)

def bimodal(x, params):
    pdf, params1, params2, p = params # unpack parameters
    return p * pdf(x, params1) + (1. - p) * pdf(x, params2)

def vonMises(x, params):
    mean, kappa = params # unpack parameters
    normalization = 1. / (np.pi * special.iv(0, kappa))
    return normalization * np.exp(kappa * np.cos(2.0 * x - mean))

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

def dust_T_pol(chi, mean_chi, X, mean_T):

    delta_chi_min = .5 * np.pi - np.abs(np.abs(chi - mean_chi) - .5 * np.pi)

    return mean_T + X * delta_chi_min / (np.pi / 2.)

def dust_Q_pol(chi, nu, mean_T, mean_chi, X=4.0):
    return B_nu(nu, dust_T_pol(chi, mean_chi, X, mean_T)) \
               * G_nu(nu_ref, Tcmb) \
               / ( B_nu(353.*1e9, dust_T_pol(chi, mean_chi, X, mean_T)) * G_nu(nu, Tcmb) ) \
                * np.cos(2. * chi) * vonMises(chi, (mean_chi, kappa))

def dust_U_pol(chi, nu, mean_T, mean_chi, X=4.0):
    return B_nu(nu, dust_T_pol(chi, mean_chi, X, mean_T)) \
               * G_nu(nu_ref, Tcmb) \
               / ( B_nu(353.*1e9, dust_T_pol(chi, mean_chi, X, mean_T)) * G_nu(nu, Tcmb) ) \
                * np.sin(2. * chi) * vonMises(chi, (mean_chi, kappa))

def prob_MBB_Q(chi, mean_chi, beta, mean_T, nu, pdf_beta, beta_params, nu_ref):
    I1_Q = lambda nu: dust_Q_pol(chi, nu, mean_T, mean_chi)
    Q_ref = integrate.simps(I1_Q(nu_ref), chi)

    int_I1_Q = [integrate.simps(I1_Q(nu), chi) / Q_ref for nu in nu]

    I2_Q = lambda nu: (nu / nu_ref)**beta * pdf_beta(beta, beta_params)

    int_I2_Q = [integrate.simps(I2_Q(nu), beta) for nu in nu]
    # print(int_I1_Q)

    return np.asarray(int_I1_Q) * np.asarray(int_I2_Q)

def prob_MBB_U(chi, mean_chi, beta, mean_T, nu, pdf_beta, beta_params, nu_ref):
    I1_U = lambda nu: dust_U_pol(chi, nu, mean_T, mean_chi)
    U_ref = integrate.simps(I1_U(nu_ref), chi)

    int_I1_U = [integrate.simps(I1_U(nu), chi) / U_ref for nu in nu]

    I2_U = lambda nu: (nu / nu_ref)**beta * pdf_beta(beta, beta_params)

    int_I2_U = [integrate.simps(I2_U(nu), beta) for nu in nu]
    # print(int_I1_U)

    return np.asarray(int_I1_U) * np.asarray(int_I2_U)
