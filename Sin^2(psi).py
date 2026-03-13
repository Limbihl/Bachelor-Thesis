import os
import numpy as np
import frm2_data_loader_sin2psi as loader
import constants as const
import fit_run  # Assuming this contains your fitting logic

# Physical Constants
LAM_K1 = const.LAM_K1  # Cr-K_alpha in Angstrom


def run_stress_analysis():
    # 1. Setup paths and file lists
    folder_path = r"C:\Users\ludig\OneDrive - TUM\Dokumente\Desktop\Studium\7 Semester\BA Arbeit\FRM 2 Dateien\FOPRA NaCl XRD Daten\NaCl Peaks (111) (200) (220) (311)"
    master_files = ["2052_00009205.dat", "2052_00009206.dat", "2052_00009207.dat"]  # 900s scans

    # 2. Load the measurement map (The "Aktenordner")
    data_dict = loader.load_and_convert_frm2_data(master_files, folder_path)

    # 3. Container for final physics results grouped by Phi
    analysis_results = {}

    # Iterate over each Phi block
    for key, block in data_dict.items():
        phi_val = block['phi']
        psi_angles = block['psi']
        csv_files = block['csv_files']

        print(f"\n--- Processing Block: {key} (Phi = {phi_val}°) ---")

        d_spacings = []
        sin2psi_values = []

        # Iterate over each measurement point within the Phi block
        for psi, csv_name in zip(psi_angles, csv_files):
            csv_full_path = os.path.join(folder_path, csv_name)

            # --- PEAK FITTING ---
            # Assuming fit_run.get_peak_position returns 2Theta in degrees
            two_theta = fit_run.get_peak_position(csv_full_path)

            if two_theta is not None:
                # --- PHYSICS CALCULATION ---
                # Bragg's Law: d = lambda / (2 * sin(theta))
                theta_rad = np.radians(two_theta / 2.0)
                d = LAM_K1 / (2.0 * np.sin(theta_rad))

                # Calculate sin^2(psi)
                sin2psi = np.sin(np.radians(psi)) ** 2

                d_spacings.append(d)
                sin2psi_values.append(sin2psi)
                print(f"  Psi={psi:4.1f}° | 2Theta={two_theta:7.3f}° | d={d:7.5f} A")

        # Store results for this Phi block
        analysis_results[key] = {
            'phi': phi_val,
            'sin2psi': np.array(sin2psi_values),
            'd_spacings': np.array(d_spacings)
        }

    return analysis_results


if __name__ == "__main__":
    final_data = run_stress_analysis()
