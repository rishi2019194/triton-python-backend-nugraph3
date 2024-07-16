import numpy as np
import h5py


with h5py.File("python_backend_scripts/NeutrinoML_rishi.h5", 'r') as test_data:
    test_inputs = {
        "hit_table_hit_id": np.array(test_data['hit_table']['hit_id']).flatten().reshape(-1).astype(np.int32),
        "hit_table_local_plane": np.array(test_data['hit_table']['local_plane']).flatten().reshape(-1).astype(np.int32),
        "hit_table_local_time": np.array(test_data['hit_table']['local_time']).flatten().reshape(-1).astype(np.float32),
        "hit_table_local_wire": np.array(test_data['hit_table']['local_wire']).flatten().reshape(-1).astype(np.int32),
        "hit_table_integral": np.array(test_data['hit_table']['integral']).flatten().reshape(-1).astype(np.float32),
        "hit_table_rms": np.array(test_data['hit_table']['rms']).flatten().reshape(-1).astype(np.float32),

        "spacepoint_table_spacepoint_id": np.array(test_data["spacepoint_table"]["spacepoint_id"]).flatten().reshape(-1).astype(np.int32),
        "spacepoint_table_hit_id_u": np.array(test_data["spacepoint_table"]["hit_id"][:,0:1]).flatten().reshape(-1).astype(np.int32),
        "spacepoint_table_hit_id_v": np.array(test_data["spacepoint_table"]["hit_id"][:,1:2]).flatten().reshape(-1).astype(np.int32),
        "spacepoint_table_hit_id_y": np.array(test_data["spacepoint_table"]["hit_id"][:,2:3]).flatten().reshape(-1).astype(np.int32)
    }


filename = 'data.txt'

with open(filename, 'w') as f:
    for key, value in test_inputs.items():
        # Write the key and the number of elements
        f.write(key + '\n')
        f.write(f"{len(value)} {' '.join(map(str, value))}\n\n")