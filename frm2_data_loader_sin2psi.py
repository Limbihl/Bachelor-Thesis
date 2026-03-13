import os
import read_raw_to_cvs  # Importing your beautifully refactored conversion script


def load_and_convert_frm2_data(dat_filenames, data_folder_path):
    """
    Parses FRM2 .dat files to extract phi and psi angles.
    Checks if the corresponding .csv files exist. If not, it converts
    the .raw files on the fly using the read_raw_to_cvs module.

    Args:
        dat_filenames (list): List of .dat filenames.
        data_folder_path (str): Directory containing the data.

    Returns:
        dict: Structured dictionary with psi angles and csv filenames grouped by phi.
    """
    measurement_data = {}

    for dat_file in dat_filenames:
        dat_path = os.path.join(data_folder_path, dat_file)

        if not os.path.exists(dat_path):
            print(f"WARNING: File not found - {dat_path}")
            continue

        phi_value = None
        psi_list = []
        csv_list = []
        is_table_section = False

        with open(dat_path, 'r', encoding='utf-8', errors='ignore') as file:
            for line in file:
                line = line.strip()

                # Extract Phi
                if line.startswith("#") and "phis_value" in line:
                    parts = line.split(":")
                    if len(parts) == 2:
                        phi_value = float(parts[1].strip().split()[0])

                # Detect table start
                if line.startswith("# chis") and "file1" in line:
                    is_table_section = True
                    continue

                # Skip units
                if is_table_section and line.startswith("# deg"):
                    continue

                # Parse rows
                if is_table_section and not line.startswith("#") and line:
                    columns = line.split(";")
                    if len(columns) >= 2:
                        psi_val = float(columns[0].strip())

                        right_side = columns[1].split()
                        raw_filename = right_side[-1].strip()
                        csv_filename = raw_filename.replace(".raw", ".csv")

                        raw_path = os.path.join(data_folder_path, raw_filename)
                        csv_path = os.path.join(data_folder_path, csv_filename)

                        # --- THE ELEGANT ON-THE-FLY CONVERSION ---
                        if not os.path.exists(csv_path):
                            if os.path.exists(raw_path):
                                print(f"Converting on the fly: {raw_filename} -> {csv_filename}")

                                # Let your helper script do 100% of the heavy lifting!
                                conversion_success = read_raw_to_cvs.convert_single_file(raw_path, csv_path)

                                if not conversion_success:
                                    print(f"ERROR: Conversion failed for {raw_filename}")
                            else:
                                print(f"ERROR: Missing raw file {raw_filename}")
                        # -----------------------------------------

                        psi_list.append(psi_val)
                        csv_list.append(csv_filename)

        # Save to our "Aktenordner"
        if phi_value is not None:
            measurement_data[phi_value] = {
                'psi': psi_list,
                'csv_files': csv_list
            }

    return measurement_data


if __name__ == "__main__":
    folder_path = r"C:\Users\ludig\OneDrive - TUM\Dokumente\Desktop\Studium\7 Semester\BA Arbeit\FRM 2 Dateien\FOPRA NaCl XRD Daten\NaCl Peaks (111) (200) (220) (311)"
    master_files = ["2052_00009202.dat", "2052_00009204.dat", "2052_00009207.dat"]

    data_dict = load_and_convert_frm2_data(master_files, folder_path)

    # Just a small check to see if our dictionary works
    print("\n--- Data Loading Complete ---")
    for phi, data in data_dict.items():
        print(f"Phi: {phi}° -> Found {len(data['psi'])} measurements.")