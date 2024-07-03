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
import numpy as np
import torch
from torch_geometric.data import Data, HeteroData

def load_heterodata(f, name: str):
    data = HeteroData()
    # Read the whole dataset idx, dataset name is self.groups[idx]
    group = f[f'dataset/{name}'][()]
    for dataset in group.dtype.names:
        store, attr = dataset.split('/')
        if "_" in store: store = tuple(store.split("_"))
        if group[dataset].ndim == 0:
            if attr == 'edge_index': # empty edge tensor
                data[store][attr] = torch.LongTensor([[],[]])
            else: # scalar
                data[store][attr] = torch.as_tensor(group[dataset][()])
        else: # multi-dimension array
            data[store][attr] = torch.as_tensor(group[dataset][:])
    return data

model_name = "nugraph3"

with h5py.File('nugraph3_data.h5', 'r') as f:
    test_samples = f['samples']['test'].asstr()[()]

    test_data = load_heterodata(f, test_samples[0])

    test_inputs = {
        
        "sp_num_nodes":np.array([(test_data['sp'].num_nodes).item()]).astype(np.float32),
        "evt_y":np.array([(test_data['evt'].y).item()]).astype(np.float32),
        "u_x_dict":test_data['u'].x.numpy().astype(np.float32),
        "u_pos":test_data['u'].pos.numpy().astype(np.float32),
        "v_x_dict":test_data['v'].x.numpy().astype(np.float32),
        "v_pos":test_data['v'].pos.numpy().astype(np.float32),
        "y_x_dict":test_data['y'].x.numpy().astype(np.float32),
        "y_pos":test_data['y'].pos.numpy().astype(np.float32),
        "u_plane_u":test_data[('u','plane','u')].edge_index.numpy().astype(np.int64),
        "u_nexus_sp":test_data[('u', 'nexus', 'sp')].edge_index.numpy().astype(np.int64),
        "v_plane_v":test_data[('v','plane','v')].edge_index.numpy().astype(np.int64),
        "v_nexus_sp":test_data[('v', 'nexus', 'sp')].edge_index.numpy().astype(np.int64),
        "y_plane_y":test_data[('y','plane','y')].edge_index.numpy().astype(np.int64),
        "y_nexus_sp":test_data[('y', 'nexus', 'sp')].edge_index.numpy().astype(np.int64)

    }


    test_inputs_triton = {}
    c = 0
    for k in test_inputs.keys():
        test_inputs_triton[k] = test_inputs[k]
        c += 1
with httpclient.InferenceServerClient("localhost:8000") as client:
    inputs = []
    for key in test_inputs_triton:
        if(key == "u_nexus_sp" or key == "u_plane_u" or key == "v_nexus_sp" or key == "v_plane_v" or key == "y_nexus_sp" or key == "y_plane_y"):
            input = httpclient.InferInput(key, test_inputs_triton[key].shape, "INT64")
        else:
            input = httpclient.InferInput(key, test_inputs_triton[key].shape, "FP32")
        input.set_data_from_numpy(test_inputs_triton[key])
        inputs.append(input)

    examples_input = np.asarray(["hi", "lo"], dtype=object)
    print(examples_input.shape)
    text = httpclient.InferInput('examples', examples_input.shape, "BYTES")
    text.set_data_from_numpy(examples_input)
    inputs.append(text)

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