import numpy as np
import pandas as pd


with pd.HDFStore('python_backend_scripts/nugraph3_raw_event.h5', 'r') as test_data:
    test_inputs = {
        "hit_table_hit_id": np.array(test_data['hit_table']['hit_id']).astype(np.int32),
        "hit_table_local_plane": np.array(test_data['hit_table']['local_plane']).astype(np.int32),
        "hit_table_local_time": np.array(test_data['hit_table']['local_time']).astype(np.float32),
        "hit_table_local_wire": np.array(test_data['hit_table']['local_wire']).astype(np.int32),
        "hit_table_integral": np.array(test_data['hit_table']['integral']).astype(np.float32),
        "hit_table_rms": np.array(test_data['hit_table']['rms']).astype(np.float32),

        "spacepoint_table_spacepoint_id": np.array(test_data["spacepoint_table"]["spacepoint_id"]).astype(np.int32),
        "spacepoint_table_hit_id_u": np.array(test_data["spacepoint_table"]["hit_id_u"]).astype(np.int32),
        "spacepoint_table_hit_id_v": np.array(test_data["spacepoint_table"]["hit_id_v"]).astype(np.int32),
        "spacepoint_table_hit_id_y": np.array(test_data["spacepoint_table"]["hit_id_y"]).astype(np.int32)
    }


filename = 'data.txt'

with open(filename, 'w') as f:
    for key, value in test_inputs.items():
        # Write the key and the number of elements
        f.write(key + '\n')
        f.write(f"{len(value)} {' '.join(map(str, value))}\n\n")