import numpy as np
from numpy import dtype
import os


class ArrayDesc:
    """Defines the properties of an array detector result.

    An array type consists of these attributes:

    * name, a name for the array
    * shape, a tuple of lengths in 1 to N dimensions arranged as for C order
      arrays, i.e. (..., t, y, x), which is also the numpy shape
    * dtype, the data type of a single value, in numpy format
    * dimnames, a list of names for each dimension

    The class can try to determine if a given image-type can be converted
    to another.
    """

    def __init__(self, name, shape, dtype, dimnames=None):
        """Creates a datatype with given (numpy) shape and (numpy) data format.

        Also stores the 'names' of the used dimensions as a list called
        dimnames.  Defaults to 'X', 'Y' for 2D data and 'X', 'Y', 'Z' for 3D
        data.
        """
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
    with open(file_path, 'rb') as f:
        c = f.read()
        hs = c.find(b'\n### NICOS Device snapshot')
        header = c[hs:].decode('utf_8')
        print(header)

    for l in header.split('\n'):
        if l.startswith('ArrayDesc'):
            a = eval(l)
            return np.fromfile(file_path,
                               a.dtype, np.prod(a.shape)).reshape(a.shape)


def write_csv_file(file_path, data_array):
    # Modify this function based on your specific requirements to write data to the CSV file
    # Example: Assume the data is written as a list of numbers in a CSV file
    with open(file_path, 'w') as file:
        # Add a header with column names
        # file.write("Index,Value\n")

        # Write data to corresponding CSV file
        for i, value in enumerate(data_array):
            calculated_value = start_value + i * interval
            file.write(f"{calculated_value},{value}\n")


def convert_raw_to_csv(folder_path):
    # Get a list of all files in the specified folder
    file_list = [f for f in os.listdir(folder_path) if f.endswith('.raw')]

    # Iterate through each file, read its data, and write it to a corresponding CSV file
    for file_name in file_list:
        raw_file_path = os.path.join(folder_path, file_name)
        csv_file_path = os.path.join(folder_path, file_name.replace('.raw', '.csv'))

        # Read data from .raw file
        data_array = read_raw_file(raw_file_path)

        # Write data to corresponding CSV file with a counter as the first column
        write_csv_file(csv_file_path, data_array)


# Specify the folder containing your .raw files
folder_path = r"C:\Users\ludig\OneDrive - TUM\Dokumente\Desktop\Studium\7 Semester\BA Arbeit\FRM 2 Dateien\RSXRD\AISI304"



# Specify the known starting value and interval


# specify tths angle of detector
tths = 128.3764

start_value = tths - 9
# 18 deg coverage, 1280 pixel line detector
interval = 18 / 1280

# Call the function to convert .raw files to .dat files
convert_raw_to_csv(folder_path)

"""
image = read_raw_file('F:/Xrd/xresd_003_00027035.raw')

print(np.sum(image))
print(image)
"""
