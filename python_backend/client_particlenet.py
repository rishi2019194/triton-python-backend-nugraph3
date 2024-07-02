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

model_name = "particlenet"

with httpclient.InferenceServerClient("localhost:8000") as client:
    for track in [100, 200]:
        print("For tracks: ", track)
        # create 5 random jets with 100 tracks each
        # data_points = 
        test_inputs = {'points': np.random.rand(5,2,track).astype(np.float32),
                    'features': np.random.rand(5,5,track).astype(np.float32),
                    'mask': np.ones((5,1,track),dtype=np.float32)}


        test_inputs_triton = {}
        c = 0
        for k in test_inputs.keys():
            test_inputs_triton[f'{k}__{c}'] = test_inputs[k]
            c += 1
        inputs = []
        for key in test_inputs_triton:
            input = httpclient.InferInput(key, test_inputs_triton[key].shape, "FP32")
            input.set_data_from_numpy(test_inputs_triton[key])
            inputs.append(input)

        outputs = [
            httpclient.InferRequestedOutput("softmax__0"),
        ]

        response = client.infer(model_name, inputs, request_id=str(track), outputs=outputs)
        result = response.get_response()
        output0_data = response.as_numpy("softmax__0")

        print("Triton output: ")
        print(output0_data)