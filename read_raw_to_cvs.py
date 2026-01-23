import numpy as np
from numpy import dtype
import os



# Specify the folder containing your .raw files
folder_path = r"C:\Users\ludig\OneDrive - TUM\Dokumente\Desktop\Studium\7 Semester\BA Arbeit\FRM 2 Dateien\FOPRA NaCl XRD Daten\NaCl Peaks (111) (200) (220) (311)"


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


def read_raw_file(file_path):
    header = ""
    tths_value = 0.0
    ysd_value = 305.0

    with open(file_path, 'rb') as f:
        c = f.read()
        header_start = c.find(b'\n### NICOS Device snapshot')
        # errors='ignore' verhindert Abstürze bei seltsamen Zeichen
        header = c[header_start:].decode('utf_8', errors='ignore')
        # print(header)


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

        #Schreibt Daten in eine CSV.

        with open(file_path, 'w') as file:
            # 1. Header hinzufügen (wichtig für Origin/Excel später)
            file.write("TwoTheta_deg,Counts\n")

            # 2. Iterieren über Winkel UND Daten gleichzeitig
            for angle, count in zip(angle_array, data_array):
                # Formatiere Winkel auf 5 Nachkommastellen (.5f)
                file.write(f"{angle:.5f},{count}\n")


# We have 1280 pixels, with the index starting at 0.
# The center of the detector lies between pixels 1279/2 and 1280/2.
# 1 Pixel = 0.05 mm. Therefore, the center is located at Y_center = 639 * d_pixel.

# Since the detector is flat and not curved, we need to convert pixels to tths_value:
# The relative displacement to the center of the detector is:
# Y_shift = (i - 0.5 - 639) * d_pixel
# ...where the -0.5 accounts for the fact that we consider the center of the respective pixel (instead of the edge).

# Then: tths_value_of_pixel_i = tths_value + arctan(Y_displacement / ysd_value)
# ...with ysd_value being the radius from the center to the detector.


def convert_raw_to_csv(folder_path):
    # Constants (Physics/Hardware)
    d_pixel = 0.05  # Width of one pixel in mm
    i_center = 639  # Pixel index of the center

    file_list = [f for f in os.listdir(folder_path) if f.endswith('.raw')]

    for file_name in file_list:
        raw_file_path = os.path.join(folder_path, file_name)
        csv_file_path = os.path.join(folder_path, file_name.replace('.raw', '.csv'))

        # 1. Read metadata and counts
        # IMPORTANT: Your read_raw_file must return (data, tths, ysd)!
        data_array, tths, ysd = read_raw_file(raw_file_path)

        if data_array is not None:
            print(f"Processing {file_name}: Center={tths}°, Distance={ysd}mm")

            # --- CALCULATION HAPPENS HERE ---

            # A. Create an array from 0 to 1279
            i = np.arange(len(data_array))

            # B. Calculate physical displacement in mm (your formula)
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




# Call the function
convert_raw_to_csv(folder_path)