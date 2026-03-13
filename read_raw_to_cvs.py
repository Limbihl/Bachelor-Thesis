import numpy as np
from numpy import dtype
import os



class ArrayDesc:
    """Defines the properties of an array detector result."""

    def __init__(self, name, shape, dtype, dimnames=None):
        self.name = name
        self.shape = shape
        self.dtype = np.dtype(dtype)
        if dimnames is None:
            dimnames = ['X', 'Y', 'Z', 'T', 'E', 'U', 'V', 'W'][:len(shape)]
        self.dimnames = dimnames

    def __repr__(self):
        return 'ArrayDesc(%r, %r, %r, %r)' % (self.name, self.shape,
                                              self.dtype, self.dimnames)

    def copy(self):
        return ArrayDesc(self.name, self.shape, self.dtype, self.dimnames)


def extract_pixel_data_and_metadata_from_raw(file_path):
    """
        Reads a binary .raw FRM2 file.
        Extracts the 1D pixel data array and the specific metadata
        (detector angle and sample distance) from the file header.

        Args:
            file_path (str): The full path to the .raw file.

        Returns:
            tuple: (data_array, tths_value, ysd_value)
                - data_array (numpy.ndarray): The 1280 intensity values (counts).
                - tths_value (float): The detector start position in degrees (2Theta).
                - ysd_value (float): The sample-detector distance in mm (radius R).
        """
    header = ""
    # Default fallback values in case the .raw header is corrupted or missing.
    tths_value = 0.0 # tths_value: Default detector start position in degrees (2Theta)
    ysd_value = 305.0 # ysd_value: Standard sample-detector distance in mm (radius R)

    with open(file_path, 'rb') as f:
        c = f.read()
        header_start = c.find(b'\n### NICOS Device snapshot')
        # errors='ignore' prevents crashes with strange characters
        header = c[header_start:].decode('utf_8', errors='ignore')

    # --- Extract Metadata from Header---
    for line in header.split('\n'):
        # Extract tths
        if "tths_value" in line and ":" in line:
            try:
                val = line.split(':')[1].strip().split()[0]
                tths_value = float(val)
            except:
                pass

        # Extract ysd (radius R)
        if "ysd_value" in line and ":" in line:
            try:
                val = line.split(':')[1].strip().split()[0]
                ysd_value = float(val)
            except:
                pass

    # --- Read Data ---
    # data is the vektor with the 1280 intensity values
    for l in header.split('\n'):
        if l.startswith('ArrayDesc'):
            a = eval(l)
            data = np.fromfile(file_path, a.dtype, np.prod(a.shape)).reshape(a.shape)
            return data, tths_value, ysd_value

    return None, 0.0, 305.0


def write_csv_file(file_path, data_array, angle_array):
    """Writes data into a CSV."""
    with open(file_path, 'w') as file:
        # 1. Add header (important for Origin/Excel later)
        file.write("TwoTheta_deg,Counts\n")

        # 2. Iterate over angles AND data simultaneously
        for angle, count in zip(angle_array, data_array):
            # Format angle to 5 decimal places (.5f)
            file.write(f"{angle:.5f},{count}\n")


def convert_single_file(raw_file_path, csv_file_path):
    """
    Reads a single .raw file, applies the geometric correction to calculate
    the true 2Theta angles, and writes the result to a .csv file.
    """
    # Constants (Physics/Hardware)
    d_pixel = 0.05  # Width of one pixel in mm
    i_center = 639  # Pixel index of the center

    # 1. Read metadata and counts
    data_array, tths, ysd = extract_pixel_data_and_metadata_from_raw(raw_file_path)

    if data_array is not None:
        print(f"Processing file: Center={tths}°, Distance={ysd}mm")

        # --- CALCULATION HAPPENS HERE ---

        # A. Create an array from 0 to 1279
        i = np.arange(len(data_array))

        # B. Calculate physical displacement in mm
        # y_shift = (Index - Pixel_Center_Correction - Hardware_Center) * Pixel_Size
        y_shift = (i - 0.5 - i_center) * d_pixel

        # C. Calculate angle offset (Geometric correction)
        # arctan( Opposite / Adjacent ) -> Result is in radians
        angle_offset_rad = np.arctan(y_shift / ysd)

        # D. Convert to degrees
        angle_offset_deg = np.degrees(angle_offset_rad)

        # E. Final Angle = Motor Center Angle + Offset
        calculated_angles = tths + angle_offset_deg

        # ------------------------------------

        # 2. Pass finished arrays for writing
        write_csv_file(csv_file_path, data_array, calculated_angles)
        return True

    return False


def convert_all_in_folder_raw_to_csv(folder_path):
    """
    Iterates over all .raw files in the specified folder and converts them.
    """
    file_list = [f for f in os.listdir(folder_path) if f.endswith('.raw')]

    for file_name in file_list:
        raw_file_path = os.path.join(folder_path, file_name)
        csv_file_path = os.path.join(folder_path, file_name.replace('.raw', '.csv'))

        # Call the new single-file function
        convert_single_file(raw_file_path, csv_file_path)


# Prevent execution when imported as a module in other scripts
if __name__ == "__main__":
    # Specify the folder containing your .raw files
    folder_path = r"C:\Users\ludig\OneDrive - TUM\Dokumente\Desktop\Studium\7 Semester\BA Arbeit\FRM 2 Dateien\FOPRA NaCl XRD Daten\NaCl Peaks (111) (200) (220) (311)"

    convert_all_in_folder_raw_to_csv(folder_path)