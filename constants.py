"""
Central configuration file for physical constants and parameters.
This ensures consistency across all analysis scripts (fitting, calculation, plotting).
"""

# --- Wavelengths (Chromium Anode) ---
# Values in Angstrom [A]
LAM_K1 = 2.28970       # Cr K-Alpha 1 wavelength
LAM_K2 = 2.29361       # Cr K-Alpha 2 wavelength
INTENSITY_RATIO = 0.5  # Intensity ratio I(Ka2) / I(Ka1)

# --- Material Data (Sodium Chloride - NaCl) ---
# Literature value for lattice parameter 'a' in Angstrom [A]
# (Source: Standard literature for NaCl at room temperature)
LIT_LATTICE_PARAM = 5.6402