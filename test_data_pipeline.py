import os
import frm2_data_loader_sin2psi as data_loader

"""2052_00009202.dat,2052_00009203.dat,2052_00009204.dat,2052_00009205.dat,2052_00009206.dat,2052_00009207.dat    """

def run_pipeline_test():
    # 1. Setup
    folder_path = r"C:\Users\ludig\OneDrive - TUM\Dokumente\Desktop\Studium\7 Semester\BA Arbeit\phi, psi scans\data"
    master_files = ["2052_00009205.dat", "2052_00009206.dat", "2052_00009207.dat"]

    print("--- 🚀 Starting Data Pipeline Test ---")

    # 2. Run the loader (this will trigger the conversion if needed)
    data_dict = data_loader.load_and_convert_frm2_data(master_files, folder_path)

    print("\n--- 📊 Checking Dictionary Structure ---")

    # 3. Analyze the results
    for phi_angle, data in data_dict.items():
        psi_list = data['psi']
        csv_list = data['csv_files']

        psi_count = len(psi_list)
        csv_count = len(csv_list)

        print(f"\n[Phi = {phi_angle}°]")
        print(f"  -> Found {psi_count} Psi angles and {csv_count} CSV files.")

        # Test A: Length match
        if psi_count != csv_count:
            print("  -> ❌ ERROR: Mismatch! Lists have different lengths.")
            continue
        else:
            print("  -> ✅ SUCCESS: Psi and CSV lists match perfectly.")

        # Test B: File existence and content check (we just test the first file of each Phi)
        if csv_count > 0:
            first_csv_name = csv_list[0]
            first_csv_path = os.path.join(folder_path, first_csv_name)

            if os.path.exists(first_csv_path):
                print(f"  -> ✅ SUCCESS: File exists on disk ({first_csv_name})")

                # Let's peek into the file to see if the header and first data row look good
                with open(first_csv_path, 'r') as file:
                    # Read the first 3 lines
                    preview_lines = [next(file).strip() for _ in range(3)]

                print("  -> 📄 File Preview (First 3 lines):")
                for line in preview_lines:
                    print(f"       {line}")
            else:
                print(f"  -> ❌ ERROR: File NOT found on disk! ({first_csv_name})")


if __name__ == "__main__":
    run_pipeline_test()