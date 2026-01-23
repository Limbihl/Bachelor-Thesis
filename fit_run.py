import xrd_math_models
import constants as const
import read_raw_to_cvs
import matplotlib.pyplot as plt
import numpy as np
import os

#CVS file path
file_path_cvs = r"C:\Users\ludig\OneDrive - TUM\Dokumente\Desktop\Studium\7 Semester\BA Arbeit\FRM 2 Dateien\FOPRA NaCl XRD Daten\NaCl Peaks (111) (200) (220) (311)\2069_00461483.csv"


def plot_fit_results(x, y, y_fit, y_ka1, y_ka2, residuals, params, fit_bounds):
    """
    Creates a professional plot showing the Data, Doublet Fit, and Residuals.

    Parameters:
    - x, y: The observed data (ROI).
    - y_fit: The total calculated model.
    - y_ka1, y_ka2: The individual split peaks (for visualization).
    - residuals: y - y_fit.
    - params: The optimized parameters [A, x0, fwhm, eta, m, b].
    - fit_bounds: Tuple (min, max) for the x-axis limits.
    """

    # Unpack parameters for the title
    A1, x01, fwhm, eta, m, b = params
    fit_min, fit_max = fit_bounds

    # Create the figure with two subplots (Top: Fit, Bottom: Residuals)
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True,
                                   gridspec_kw={'height_ratios': [3, 1]},
                                   figsize=(8, 6))

    # --- Top Plot: Data & Fit ---
    ax1.scatter(x, y, s=15, color='black', alpha=0.6, label='ROI Data')
    ax1.plot(x, y_fit, color='red', linewidth=2, label='Doublet Fit')

    # Plot individual components (dashed lines)
    ax1.plot(x, y_ka1, color='green', linestyle='--', linewidth=1, alpha=0.7, label='Kα1')
    ax1.plot(x, y_ka2, color='blue', linestyle='--', linewidth=1, alpha=0.7, label='Kα2')

    ax1.set_ylabel("Intensity [Counts]")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f"Fit Result (x0={x01:.3f}°, FWHM={fwhm:.3f}°)")

    # --- Bottom Plot: Residuals ---
    ax2.scatter(x, residuals, s=10, color='blue', alpha=0.6)
    ax2.axhline(0, color='black', linestyle='-', linewidth=1)  # Zero line
    ax2.set_ylabel("Residuals")
    ax2.set_xlabel("2Theta [deg]")
    ax2.grid(True, alpha=0.3)

    # Layout adjustments
    plt.xlim(fit_min - 0.2, fit_max + 0.2)  # Zoom to ROI + margin
    plt.subplots_adjust(hspace=0.05)  # Reduce space between plots

    # Important: block=True ensures the script pauses until you close the window
    print("--> Displaying plot. Close window to continue...")
    plt.show(block=True)



def load_csv(file_path):
    angle, intensity = np.loadtxt(
        file_path, delimiter=",", unpack=True, skiprows=1
    )
    return angle, intensity


angle, intensity = load_csv(file_path_cvs)



def preview_data(x, y, filename="Data Preview"):
    """
    Plots the raw measurement data and pauses execution until the window is closed.
    """
    print(f"--> Showing preview for: {filename}")
    print("    [PAUSED] Please close the plot window to continue...")

    plt.figure(figsize=(10, 6))

    # Plot as scatter points (better for seeing actual measurement density)
    plt.scatter(x, y, s=2, color='black', alpha=0.6, label="Raw Data")

    # Alternative: use plt.plot(x, y) if you prefer a connected line

    plt.title(f"Preview: {filename}")
    plt.xlabel("2Theta [deg]")
    plt.ylabel("Intensity [Counts]")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show(block=True) # This command blocks the script until the window is closed!

preview_data(angle, intensity, filename=os.path.basename(file_path_cvs))



def determine_roi_from_header(csv_file_path, window):
    """
    Reads the corresponding .raw file header to find the motor position (tths).
    Returns the scan center and the calculated fit bounds.
    """
    # Construct path to the corresponding .raw file
    file_path_raw = csv_file_path.replace('.csv', '.raw')

    try:
        # Read the header to get the motor position (tths_value)
        _, tths_header, _ = read_raw_to_cvs.read_raw_file(file_path_raw)

        print(f"--> Header Info: Motor position was at {tths_header}°")

        # Calculate bounds
        fit_window_min = tths_header - window
        fit_window_max = tths_header + window

        return fit_window_min, fit_window_max, tths_header

    except FileNotFoundError:
        print(f"CRITICAL ERROR: Could not find raw file at {file_path_raw}")
        print("Cannot determine scan center from header.")
        exit()
    except Exception as e:
        print(f"CRITICAL ERROR reading header: {e}")
        exit()



# ==========================================
# 2. MASKING & SLICING
# ==========================================
fit_window_min, fit_window_max, tths_header = determine_roi_from_header(file_path_cvs, window=1)
print(f"--> Slicing data to range: {fit_window_min:.2f}° - {fit_window_max:.2f}°")

# Create mask based on the calculated fit_min/fit_max
mask = (angle >= fit_window_min) & (angle <= fit_window_max)
# Slice/Crop data based on the mask
x = angle[mask]
y = intensity[mask]


# --- SAFETY CHECK ---
if len(x) == 0:
    print(f"CRITICAL ERROR: No data points found in the range {fit_window_min}° - {fit_window_max}°!")
    print(f"The peak in this file is likely at a different position (check the Header Info printed above).")
    print("Please adjust 'fit_min' and 'fit_max' in your script.")
    exit() # Stops the script cleanly before it crashes
# --------------------



LAM_K1 = const.LAM_K1 # Å, Cr Kα1
LAM_K2 = const.LAM_K2  # Å, Cr Kα2
RATIO  = const.INTENSITY_RATIO  # I(Kα2)/I(Kα1)


#initial guess

x0_guess_default = tths_header # use the tths_header position as intitial guess for x0


A_guess= 600 # Amplitude (its the total amplitude - the background)
x0_guess= x0_guess_default  # Peak position
fwhm_guess= 0.2   # Full Width at Half Maximum
eta_guess = 0.5 # superposition parameter (eta=1 -> only Lorentz)
m_guess = -0.1  # linear background slope
b_guess = 1000  # background offset

initial_guess_doublet= [A_guess, x0_guess, fwhm_guess, eta_guess, m_guess, b_guess]


# 4. FITTING
# ---------------------------------------------------------
dx = np.median(np.diff(x))
fwhm_min = 3*dx

bounds = [
    (0, A_guess*1.5),        # A > 0
    (fit_window_min, fit_window_max), # x0 must be in ROI
    (fwhm_min, 2.0),    # FWHM reasonable range
    (0, 1),             # eta [0, 1]
    (-np.inf, np.inf),  # m
    (0, np.inf),        # b (usually positive)
]

print("Starting fit...")
result = xrd_math_models.fit_doublet_peaks(x, y, LAM_K1, LAM_K2, RATIO, initial_guess_doublet, bounds)
params_opt = result.x

# ---------------------------------------------------------
# 5. CALCULATION OF RESULTS & CURVES
# ---------------------------------------------------------
A1, x01, fwhm, eta, m, b = params_opt

# 1. Calculate the full model curve with the optimal parameters
y_fit = xrd_math_models.doublet_model_f(params_opt, x, LAM_K1, LAM_K2, RATIO)

# 2. Calculate Residuals (Observed - Model)
residuals = y - y_fit

# 3. Calculate individual components for visualization
#    (We need to calculate x02 physically to draw the blue line correctly)
theta1 = np.deg2rad(x01 / 2.0)
theta2 = np.arcsin(np.sin(theta1) * (LAM_K2 / LAM_K1))
x02 = np.rad2deg(2*theta2)


y_ka1_only = xrd_math_models.pseudo_voigt_f(x, A1, x01, fwhm, eta) + xrd_math_models.background_f(x,m,b)
y_ka2_only = xrd_math_models.pseudo_voigt_f(x, A1 * RATIO, x02, fwhm, eta) + xrd_math_models.background_f(x,m,b)

# 4. Print Statistics
rmse = np.sqrt(np.mean(residuals**2))
print("-" * 40)
print(f"Optimization Success: {result.success}")
print(f"Optimal Params [A, x0, fwhm, eta, m, b]: {np.round(params_opt, 4)}")
print(f"RMSE (Average Error): {rmse:.4f}")
print("-" * 40)

# ---------------------------------------------------------
# 6. PLOTTING
# ---------------------------------------------------------
# Now we just call the function!
plot_fit_results(
    x, y, y_fit, y_ka1_only, y_ka2_only, residuals,
    params=params_opt,
    fit_bounds=(fit_window_min, fit_window_max)
)

