import h5py
import numpy as np

# Open the HDF5 file
file_path = 'NeutrinoML_rishi.h5'
with h5py.File(file_path, 'r') as file:
    # List all groups and datasets at the root level
    print(file.keys())
    print(np.array(file['hit_table']['hit_id']).flatten().reshape(-1))
    print(np.array(file['spacepoint_table']['hit_id'][:,0:1]).flatten().reshape(-1))
    print(np.array(file['spacepoint_table']['hit_id'][:,1:2]).flatten().reshape(-1))
    print(np.array(file['spacepoint_table']['hit_id'][:,2:3]).flatten().reshape(-1))