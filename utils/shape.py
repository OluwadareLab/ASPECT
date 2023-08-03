import numpy as np

# Specify the path to the .npy file
npy_file_path = "/aspect/ASPECT/data/encoded_seqs.npy"

# Load the .npy file using NumPy
data = np.load(npy_file_path, allow_pickle=True)

# Retrieve the dimensions of the loaded data
dimensions = data.shape

# Print the dimensions
print("Dimensions:", dimensions)
