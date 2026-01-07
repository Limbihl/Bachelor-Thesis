import xrd_models
import matplotlib.pyplot as plt
import numpy as np

file_path =     r"C:\Users\ludig\OneDrive - TUM\Dokumente\Desktop\Studium\7 Semester\BA Arbeit\FRM 2 Dateien\RSXRD\AISI304\1991_00441082.csv"

def load_csv(file_path):
    angle, intensity = np.loadtxt(
        file_path, delimiter=",", unpack=True, skiprows=1
    )
    return angle, intensity


angle, intensity = load_csv(file_path)
x = angle
y = intensity


lam_k1 = 2.28970  # Å, Cr Kα1
lam_k2 = 2.29361  # Å, Cr Kα2
ratio  = 0.5      # I(Kα2)/I(Kα1)

initial_guess_doublet= [850, 128.5 , 2, 0.5, 0.1, 625]
x01_deg_guess = initial_guess_doublet[1]






"""" initial_guess_n_peaks = [A, x0, fwhm, eta, m, b] mit A= höhe, x0= peak position, fwhm= Full Width at Half Maximum,
 eta = superposition parameter (eta=1 -> only Lorentz), m = linear background slope, b = background offset
 initial_guess_doublet=[A1, x01_deg, fwhm, eta, m, b]"""

dx = np.median(np.diff(x))
fwhm_min = 3*dx # we set a lower bound so it does not collapse to a delta function

bounds = [
    (0, 6000),  # A (Höhe)
    (x.min(), x.max()),  # x0 (Peakposition) has to be in the range of the measurement data
    (fwhm_min, np.inf),  # fwhm > 0.1
    (0, 1),  # eta  has to be inbetween 0 and 1
    (-np.inf, np.inf),  # m no restrictions
    (-np.inf, np.inf),  # b no restrictions
]

result = xrd_models.fit_doublet_peaks(x, y, lam_k1,lam_k2, ratio,  initial_guess_doublet,bounds )
params_opt= result.x


A1, x01_deg, fwhm, eta, m, b = params_opt
theta1 = np.deg2rad(x01_deg / 2.0)
theta2 = np.arcsin(np.sin(theta1)*(lam_k2/lam_k1))   # 2θ(Kα2)
x02_deg = np.rad2deg(2*theta2)
y1  = xrd_models.pseudo_voigt_f(x, A1,       x01_deg, fwhm, eta)
y2 = xrd_models.pseudo_voigt_f(x, A1*ratio, x02_deg, fwhm, eta)
y_fit = xrd_models.doublet_model_f(params_opt, x, lam_k1, lam_k2, ratio)


plt.figure()
plt.scatter(x, y, s=6, label="data")
plt.plot(x, y_fit, color ='red', lw=2, label="fit")
plt.plot(x, y1, "--", label="ka1 (only)" )
plt.plot(x, y2, "--", label="Kα2 (only)")
plt.legend(); plt.grid(True); plt.show()



print("The optimal parameters [A1, x01_deg, fwhm, eta, m, b] are", params_opt)

Total_Error= np.sqrt(xrd_models.cost_doublet(params_opt, x, y, lam_k1, lam_k2, ratio))
Average_Error = Total_Error/len(x)

print("The average error is:", Average_Error )