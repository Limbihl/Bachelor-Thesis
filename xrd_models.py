import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import optimize

"""
XRD peak fitting using a pseudo-Voigt profile.

The pseudo-Voigt function approximates the Voigt profile (Gaussian–
Lorentzian convolution) as a linear combination of Gaussian and
Lorentzian components. The fit quality is evaluated via the root mean
square error (RMSE), which is minimized using scipy.optimize.
"""


def gaussian_f(x, A, x0, sigma):
    return A * np.exp(-0.5 * ((x - x0) / sigma)**2)

def lorentzian_f(x, A, x0, gamma):
    return A * (gamma**2) / ((x - x0)**2 + gamma**2)

def pseudo_voigt_f(x, A, x0, fwhm, eta):
    # fwhm -> sigma, gamma
    sigma = fwhm / (2*np.sqrt(2*np.log(2)))  # Gaussian sigma
    gamma = fwhm / 2.0                       # Lorentzian HWHM
    return eta * lorentzian_f(x, A, x0, gamma) + (1 - eta) * gaussian_f(x, A, x0, sigma)

def background_f(x, m, b):
    return m*x + b

def fit_f(x, params, n_peaks=1):
    # for fitting n Peaks without physical constrains
    params = np.asarray(params, dtype=float)
    fit_f = np.zeros_like(x, dtype=float)

    for k in range(n_peaks):
        A, x0, fwhm, eta = params[4*k:4*k+4]
        fit_f += pseudo_voigt_f(x, A, x0, fwhm, eta)

    m, b = params[4*n_peaks:4*n_peaks+2]
    fit_f += background_f(x, m, b)
    return fit_f


def doublet_model_f(params, x, lam_k1, lam_k2, ratio):

    A1, x01_deg, fwhm, eta, m, b = params
    theta1 = np.deg2rad(x01_deg / 2.0)
    theta2 = np.arcsin(np.sin(theta1)*(lam_k2/lam_k1))   # relates the peak position of the lam_k1 peak to the lam_k2 peak
    x02_deg = np.rad2deg(2*theta2)

    fit1_f = pseudo_voigt_f(x, A1, x01_deg, fwhm, eta)
    fit2_f = pseudo_voigt_f(x, A1 * ratio, x02_deg, fwhm, eta) # implements the physical constrain: Intensity of Peak2 = ratio* Peak1
    return fit1_f + fit2_f + background_f(x, m, b)



"""Our measure of error (cost) is the mean square error"""

def cost_doublet(params, x, y, lam_k1, lam_k2, ratio):
    residuals= y - doublet_model_f(params, x, lam_k1, lam_k2, ratio)
    n = len(x)
    cost_val = (1/n)*np.sum(residuals**2)
    return cost_val

def cost_n_peaks(params, x , y, n_peaks=1 ):
    residuals= y - fit_f(x, params,n_peaks)
    n = len(x)
    cost_val = (1/n)*np.sum(residuals**2)
    return cost_val


"""optimizes the initial guess by minimizing the squared error"""

def fit_n_peaks_f(x, y, initial_guess, bounds, n_peaks=1):
    result = optimize.minimize(
        cost_n_peaks(), initial_guess,
        args=(x, y, n_peaks),
        bounds=bounds
    )
    return result


def fit_doublet_peaks(x, y, lam_k1, lam_k2, ratio, initial_guess, bounds):

    result = optimize.minimize(
        cost_doublet, initial_guess,
        args=(x, y, lam_k1, lam_k2, ratio),
        bounds=bounds
    )
    return result














