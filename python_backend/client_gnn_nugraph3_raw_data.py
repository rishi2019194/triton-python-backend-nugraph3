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

import sys

import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import *
import h5py
import pandas as pd
import torch
from torch_geometric.data import Data, HeteroData

model_name = "nugraph3_new"

with pd.HDFStore('nugraph3_raw_event.h5', 'r') as test_data:
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


    test_inputs_triton = {}
    c = 0
    for k in test_inputs.keys():
        test_inputs_triton[k] = test_inputs[k]
        c += 1
     
with httpclient.InferenceServerClient("localhost:8000") as client:
    inputs = []
    for key in test_inputs_triton:

        if("id" in key or "wire_pos" in key or key == "event_table_is_cc" or key == "event_table_nu_pdg" or \
           key == "hit_table_local_plane" or key == "hit_table_local_wire" or key == "particle_table_type"):
            input = httpclient.InferInput(key, test_inputs_triton[key].shape, "INT32")
        elif("process" in key):
            input = httpclient.InferInput(key, test_inputs_triton[key].shape, "BYTES")
        else:
            input = httpclient.InferInput(key, test_inputs_triton[key].shape, "FP32")
        input.set_data_from_numpy(test_inputs_triton[key])
        inputs.append(input)

    outputs = [
        httpclient.InferRequestedOutput("e_evt"),
        httpclient.InferRequestedOutput("x_semantic_u"),
        httpclient.InferRequestedOutput("x_semantic_v"),
        httpclient.InferRequestedOutput("x_semantic_y"),
        httpclient.InferRequestedOutput("x_filter_u"),
        httpclient.InferRequestedOutput("x_filter_v"),
        httpclient.InferRequestedOutput("x_filter_y"),
        httpclient.InferRequestedOutput("v_evt")
        
    ]

    response = client.infer(model_name, inputs, request_id=str(1), outputs=outputs)
    result = response.get_response()
    output0_data = response.as_numpy("e_evt")
    output_1_data = response.as_numpy("x_semantic_u")
    output_2_data = response.as_numpy("x_semantic_v")
    output_3_data = response.as_numpy("x_semantic_y")
    output_4_data = response.as_numpy("x_filter_u")
    output_5_data = response.as_numpy("x_filter_v")
    output_6_data = response.as_numpy("x_filter_y")
    output_7_data = response.as_numpy("v_evt")

    print("Triton output: ")
    print("e_evt: ", output0_data)
    print("x_semantic_u: ", output_1_data)
    print("x_semantic_v: ", output_2_data)
    print("x_semantic_y: ", output_3_data)
    print("x_filter_u: ", output_4_data)
    print("x_filter_v: ", output_5_data)
    print("x_filter_y: ", output_6_data)
    print("v_evt: ", output_7_data)