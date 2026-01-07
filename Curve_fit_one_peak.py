import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import optimize

file_path = r"C:\Users\ludig\OneDrive - TUM\Dokumente\Desktop\Studium\7 Semester\BA Arbeit\FRM 2 Dateien\RSXRD\AISI304\1991_00441084.csv"

def load_csv(file_path):
    angle, intensity = np.loadtxt(
        file_path, delimiter=",", unpack=True, skiprows=1
    )
    return angle, intensity

def plot_data(angle, intensity, label=None):
    plt.figure(figsize=(15, 5))
    plt.scatter(angle, intensity, s=4, label=label)
    plt.xlabel("2θ (Grad)")
    plt.ylabel("Intensität (Counts)")
    plt.grid(True)
    if label:
        plt.legend()
    plt.show()


angle, intensity = load_csv(file_path)
plot_data(angle, intensity, label="XRD")

x = angle
y = intensity


# now we have to do peak fitting: in theory ht best function to fit would be the convultion of a gaussian and a Lorentz-function. However, easier: Pseudo-Voigt
#pV(x)=ηL(x)+(1−η)G(x)
# we will use the mean square root as a measure of the "improvement" we wil call this function "cost" and try to minimize it with the help of scipy.optimize




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

def fit_f(x, params):
    A, x0, fwhm, eta, m, b = params
    return (
        pseudo_voigt_f(x, A, x0, fwhm, eta)
        + background_f(x, m, b)
    )


def cost(params, x , y ):
    residual= y - fit_f(x, params)
    n = len(x)
    cost_val = (1/n)*np.sum(residual**2)
    return cost_val

# the initial guess is really important to find the GLOBAL minimum

initial_guess = [1250, 128.5 , 2, 0.5, 0.01, 625] # initial_guess = [A, x0, fwhm, eta, m, b] mit A= höhe, x0= peak position, fwhm= Full Width at Half Maximum,
                                                  # eta = superposition parameter (eta=1 -> only Lorentz), m = linear background slope, b = background offset

# Parameter bounds
bounds = [
    (0, np.inf),  # A (Höhe) > 0
    (x.min(), x.max()),  # x0 (Peakposition) has to be in the range of the measurement data
    (1e-4, np.inf),  # fwhm > 0
    (0, 1),  # eta  has to be inbetween 0 and 1
    (-np.inf, np.inf),  # m no restrictions
    (-np.inf, np.inf),  # b no restrictions
]

result = optimize.minimize(cost, initial_guess,args=(x, y), bounds=bounds)

params_opt = result.x
print('steps:',result.nit,"average error:",np.sqrt(result.fun))
print("optimal parameters:",params_opt)

y_model = fit_f(x, result.x)

plt.figure(figsize=(15,5))
plt.scatter(x, y, s=4, label="data")
plt.plot(x, y_model, linewidth=2, label="fit")
plt.legend()
plt.grid(True)
plt.show()
