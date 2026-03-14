import sin2psi_analysis


def run_stress_analysis(folder_path, master_files, fit_window=1.0, d0=None):
    return sin2psi_analysis.analyze_measurements(
        data_folder_path=folder_path,
        dat_filenames=master_files,
        fit_window=fit_window,
        d0=d0,
    )


if __name__ == "__main__":
    folder_path = (
        r"C:\Users\ludig\OneDrive - TUM\Dokumente\Desktop\Studium\7 Semester\BA Arbeit\FRM 2 Dateien"
        r"\FOPRA NaCl XRD Daten\NaCl Peaks (111) (200) (220) (311)"
    )
    master_files = ["2052_00009205.dat", "2052_00009206.dat", "2052_00009207.dat"]

    final_data = run_stress_analysis(folder_path, master_files, fit_window=1.0)
    sin2psi_analysis.plot_metric_vs_sin2psi(final_data, metric="d_spacing_angstrom")
