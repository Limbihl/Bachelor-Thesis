import constants as const
import numpy as np


def calculate_lattice_parameter(two_theta_deg, h, k, l, wavelength):
    """
    Calculates the lattice parameter 'a' for a cubic system.

    Parameters:
    - two_theta_deg: The peak position in degrees (from the fit)
    - h, k, l: The Miller indices of the reflection
    - wavelength: Wavelength in Angstroms
    """

    # 1. Convert degrees to radians
    # IMPORTANT: We need theta, not 2theta!
    theta_rad = np.deg2rad(two_theta_deg / 2.0)

    # 2. Interplanar spacing d (Bragg's Law: lambda = 2d sin(theta))
    d_spacing = wavelength / (2 * np.sin(theta_rad))

    # 3. Lattice parameter a for cubic systems
    # Formula: a = d * sqrt(h^2 + k^2 + l^2)
    N = h ** 2 + k ** 2 + l ** 2
    a = d_spacing * np.sqrt(N)

    return a, d_spacing


# ==========================================
# ENTER YOUR VALUES HERE
# ==========================================

# Constants (Chromium K-Alpha 1)
LAM_K1 = const.LAM_K1


# Result from your fit script (copy/paste from console)
peak_position = 47.91236  # Your x01_deg value

# Which reflection is this? (Indexing)
# We know: At ~47.9 deg lies the (200) peak for NaCl/Cr
h_in = 2
k_in = 0
l_in = 0

# ==========================================
# CALCULATION & OUTPUT
# ==========================================

a_result, d_result = calculate_lattice_parameter(peak_position, h_in, k_in, l_in, LAM_K1)

print("-" * 40)
print(f"INPUT:")
print(f"  Peak Position (2Theta): {peak_position:.5f} deg")
print(f"  Reflection (hkl):       ({h_in} {k_in} {l_in})")
print("-" * 40)
print(f"RESULTS:")
print(f"  Interplanar spacing d:  {d_result:.5f} A")
print(f"  Lattice parameter a:    {a_result:.5f} A")
print("-" * 40)

# Optional: Comparison with literature
lit_val = 5.6402
deviation = abs(a_result - lit_val) / lit_val * 100
print(f"  Deviation from Lit.:    {deviation:.4f} %")
print("-" * 40)