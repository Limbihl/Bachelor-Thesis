import xrd_models
import constants as const
import read_raw_to_cvs
import matplotlib.pyplot as plt
import numpy as np


file_path =     r"C:\Users\ludig\OneDrive - TUM\Dokumente\Desktop\Testordner\2069_00461481.csv"

def load_csv(file_path):
    angle, intensity = np.loadtxt(
        file_path, delimiter=",", unpack=True, skiprows=1
    )
    return angle, intensity


angle, intensity = load_csv(file_path)

# ADJUST FITTING WINDOW (Region of Interest - ROI)
# ---------------------------------------------------------
fit_min = 47.0  # Lower bound in degrees (2Theta)
fit_max = 49.0  # Upper bound in degrees (2Theta)

# Create mask: Determine which data points lie within the window
mask = (angle >= fit_min) & (angle <= fit_max)

# Slice/Crop data based on the mask
x = angle[mask]
y = intensity[mask]



LAM_K1 = const.LAM_K1 # Å, Cr Kα1
LAM_K2 = const.LAM_K2  # Å, Cr Kα2
RATIO  = const.INTENSITY_RATIO  # I(Kα2)/I(Kα1)



# Try to get tths from the corresponding RAW file
raw_file_path = file_path.replace('.csv', '.raw')
try:
    # We ignore data (_) and ysd (_), we only want tths
    _, tths_raw, _ = read_raw_to_cvs.read_raw_file(raw_file_path)
    print(f"Initial Guess from Header: {tths_raw}°")
    x0_guess_default = tths_raw
except:
    print("Could not read RAW header, using fallback.")
    x0_guess_default = (fit_min + fit_max) / 2  # Fallback: Center of window

#initial guess
A_guess= 22000 # Amplitude
x0_guess= x0_guess_default  # Peak position
fwhm_guess= 0.2   # Full Width at Half Maximum
eta_guess = 0.2 # superposition parameter (eta=1 -> only Lorentz)
m_guess = 0   # linear background slope
b_guess = 2000  # background offset

initial_guess_doublet= [A_guess, x0_guess, fwhm_guess, eta_guess, m_guess, b_guess]


# 4. FITTING
# ---------------------------------------------------------
dx = np.median(np.diff(x))
fwhm_min = 3*dx

bounds = [
    (0, np.inf),        # A > 0
    (fit_min, fit_max), # x0 must be in ROI
    (fwhm_min, 2.0),    # FWHM reasonable range
    (0, 1),             # eta [0, 1]
    (-np.inf, np.inf),  # m
    (0, np.inf),        # b (usually positive)
]

print("Starting fit...")
result = xrd_models.fit_doublet_peaks(x, y, LAM_K1, LAM_K2, RATIO, initial_guess_doublet, bounds)
params_opt = result.x

# ---------------------------------------------------------
# 5. CALCULATION OF RESULTS & RESIDUALS
# ---------------------------------------------------------
# Recalculate the model with optimal parameters
y_fit = xrd_models.doublet_model_f(params_opt, x, LAM_K1, LAM_K2, RATIO)

# CALCULATE RESIDUALS (Messung - Modell)
residuals = y - y_fit

# Calculate individual components for plotting (optional)
A1, x01, fwhm, eta, m, b = params_opt
theta1 = np.deg2rad(x01 / 2.0)
theta2 = np.arcsin(np.sin(theta1) * (LAM_K2 / LAM_K1))
x02 = np.rad2deg(2*theta2)
background = m*x + b
y_ka1 = xrd_models.pseudo_voigt_f(x, A1, x01, fwhm, eta) + background
y_ka2 = xrd_models.pseudo_voigt_f(x, A1*RATIO, x02, fwhm, eta) + background

# Error Calculation
rmse = np.sqrt(np.mean(residuals**2))
print(f"Optimization Success: {result.success}")
print(f"Params [A, x0, fwhm, eta, m, b]: {np.round(params_opt, 4)}")
print(f"RMSE (Average Error): {rmse:.4f}")

# ---------------------------------------------------------
# 6. PROFESSIONAL PLOTTING (Residual Plot)
# ---------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True,
                               gridspec_kw={'height_ratios': [3, 1]},
                               figsize=(8, 6))

# --- Top Plot: Data & Fit ---

# Plot ROI data in black
ax1.scatter(x, y, s=15, color='black', alpha=0.6, label='ROI Data')
# Plot Fit
ax1.plot(x, y_fit, color='red', linewidth=2, label='Doublet Fit')
# Plot components (dashed)
ax1.plot(x, y_ka1, color='green', linestyle='--', linewidth=1, alpha=0.7, label='Kα1')
ax1.plot(x, y_ka2, color='blue', linestyle='--', linewidth=1, alpha=0.7, label='Kα2')

ax1.set_ylabel("Intensity [Counts]")
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_title(f"Fit Result (x0={x01:.3f}°, FWHM={fwhm:.3f}°)")

# --- Bottom Plot: Residuals ---
ax2.scatter(x, residuals, s=10, color='blue', alpha=0.6)
ax2.axhline(0, color='black', linestyle='-', linewidth=1) # Zero line
ax2.set_ylabel("Residuals")
ax2.set_xlabel("2Theta [deg]")
ax2.grid(True, alpha=0.3)

# Layout adjustments
plt.xlim(fit_min - 0.2, fit_max + 0.2) # Zoom to ROI + a bit margin
plt.subplots_adjust(hspace=0.05) # Less space between plots
plt.show()

