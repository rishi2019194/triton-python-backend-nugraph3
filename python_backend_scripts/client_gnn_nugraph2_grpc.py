# Copyright 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import *
import h5py

model_name = "nugraph2"

with h5py.File("NeutrinoML_rishi.h5", 'r') as test_data:
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


    test_inputs_triton = {}
    c = 0
    for k in test_inputs.keys():
        test_inputs_triton[k] = test_inputs[k]
        c += 1
     
with grpcclient.InferenceServerClient(url="triton.fnal.gov:443", ssl=True) as client:
    inputs = []
    for key in test_inputs_triton:

        if("id" in key or "wire_pos" in key or key == "event_table_is_cc" or key == "event_table_nu_pdg" or \
           key == "hit_table_local_plane" or key == "hit_table_local_wire" or key == "particle_table_type"):
            input = grpcclient.InferInput(key, test_inputs_triton[key].shape, "INT32")
        elif("process" in key):
            input = grpcclient.InferInput(key, test_inputs_triton[key].shape, "BYTES")
        else:
            input = grpcclient.InferInput(key, test_inputs_triton[key].shape, "FP32")
        input.set_data_from_numpy(test_inputs_triton[key])
        inputs.append(input)

    outputs = [
        grpcclient.InferRequestedOutput("x_semantic_u"),
        grpcclient.InferRequestedOutput("x_semantic_v"),
        grpcclient.InferRequestedOutput("x_semantic_y"),
        grpcclient.InferRequestedOutput("x_filter_u"),
        grpcclient.InferRequestedOutput("x_filter_v"),
        grpcclient.InferRequestedOutput("x_filter_y")
        
    ]

    response = client.infer(model_name, inputs, request_id=str(1), outputs=outputs)
    result = response.get_response()
    output_1_data = response.as_numpy("x_semantic_u")
    output_2_data = response.as_numpy("x_semantic_v")
    output_3_data = response.as_numpy("x_semantic_y")
    output_4_data = response.as_numpy("x_filter_u")
    output_5_data = response.as_numpy("x_filter_v")
    output_6_data = response.as_numpy("x_filter_y")

    print("Triton output: ")
    print("x_semantic_u: ", output_1_data)
    print("x_semantic_v: ", output_2_data)
    print("x_semantic_y: ", output_3_data)
    print("x_filter_u: ", output_4_data)
    print("x_filter_v: ", output_5_data)
    print("x_filter_y: ", output_6_data)