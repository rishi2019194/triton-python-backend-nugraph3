# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

import json
import time 
import pytorch_lightning as pl
import nugraph as ng
import torch
import numpy as np
from torch import nn
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from torch_geometric.data import HeteroData, Batch, Dataset
# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import triton_python_backend_utils as pb_utils

class HeteroDataset(Dataset):
    def __init__(self, hetero_batch, transform=None):
        super().__init__(transform=transform)
        self.hetero_batch = hetero_batch

    def len(self):
        return 1
    def get(self, idx=0):
        return self.hetero_batch
class NuGraph3_model(nn.Module):
    """
    Simple AddSub network in PyTorch. This network outpxuts the sum and
    subtraction of the inputs.
    """

    def __init__(self):
        super(NuGraph3_model, self).__init__()
        self.MODEL = ng.models.nugraph3.nugraph3.NuGraph3
        self.model = self.MODEL.load_from_checkpoint("gnn_models_check/nugraph3/1/hierarchical.ckpt", map_location='cpu')
        self.accelerator, self.devices = ng.util.configure_device()
        self.trainer = pl.Trainer(accelerator=self.accelerator, devices=self.devices,
                            logger=False)
        self.planes = ['u', 'v', 'y']
        self.norm = {'u':torch.tensor(np.array([[389.42752, 172.90794, 147.81108, 4.5563765], [147.1627, 78.01324, 228.31424, 2.2156637]]).astype(np.float32)),
                     'v':torch.tensor(np.array([[368.83023, 173.01247, 154.14513, 4.449338 ], [145.29645, 80.54078, 282.34027, 1.8969047]]).astype(np.float32)),
                     'y':torch.tensor(np.array([[546.2973, 172.77615, 116.974, 4.1647816],[283.47656, 73.99135, 115.49256, 1.4615369]]).astype(np.float32))}

    def forward(self, sp_num_nodes, u_x_dict, u_pos, v_x_dict, v_pos, y_x_dict, y_pos, evt_y, \
                u_plane_u, u_nexus_sp, v_plane_v, v_nexus_sp, y_plane_y, y_nexus_sp):
        
        sp_num_nodes = int(sp_num_nodes[0])
        evt_y = int(evt_y[0])
        sp_num_nodes = torch.tensor(sp_num_nodes)

        u_x_dict = torch.tensor(u_x_dict)
        v_x_dict = torch.tensor(v_x_dict)
        y_x_dict = torch.tensor(y_x_dict)

        u_pos = torch.tensor(u_pos)
        v_pos = torch.tensor(v_pos)
        y_pos = torch.tensor(y_pos)

        u_plane_u = torch.tensor(u_plane_u)
        u_nexus_sp = torch.tensor(u_nexus_sp)
        v_plane_v = torch.tensor(v_plane_v)
        v_nexus_sp = torch.tensor(v_nexus_sp)
        y_plane_y = torch.tensor(y_plane_y)
        y_nexus_sp = torch.tensor(y_nexus_sp)


        hetero_batch = self.create_heteroBatch(sp_num_nodes, u_x_dict, u_pos, v_x_dict, v_pos, y_x_dict, y_pos, evt_y, \
                u_plane_u, u_nexus_sp, v_plane_v, v_nexus_sp, y_plane_y, y_nexus_sp)

        transform = Compose((ng.util.PositionFeatures(self.planes),
                             ng.util.FeatureNorm(self.planes, self.norm),
                             ng.util.HierarchicalEdges(self.planes),
                             ng.util.EventLabels()))
        hetero_batch_loader = DataLoader(HeteroDataset(hetero_batch, transform=transform), batch_size=1)
        x = self.trainer.predict(self.model, hetero_batch_loader)
        return x[0]['e']['evt'].detach().numpy(), x[0]['x_semantic']['u'].detach().numpy(), \
                x[0]['x_semantic']['v'].detach().numpy(), x[0]['x_semantic']['y'].detach().numpy(), \
                x[0]['x_filter']['u'].detach().numpy(), x[0]['x_filter']['v'].detach().numpy(), \
                x[0]['x_filter']['y'].detach().numpy(), x[0]['v']['evt'].detach().numpy()
    
    def create_heteroBatch(self, sp_num_nodes, u_x_dict, u_pos, v_x_dict, v_pos, y_x_dict, y_pos, evt_y, \
                u_plane_u, u_nexus_sp, v_plane_v, v_nexus_sp, y_plane_y, y_nexus_sp):
        
        output_dict = {
            "metadata":
                {"run":torch.tensor([1]),
                 "subrun":torch.tensor([1]),
                 "event":torch.tensor([1])},
            "sp":
                {"num_nodes":sp_num_nodes},
            "u":
                {"x":u_x_dict,
                 "pos":u_pos},
            "v":
                {"x":v_x_dict,
                 "pos":v_pos},
            "y":
                {"x":y_x_dict,
                 "pos":y_pos},
            "evt":
                {"y":evt_y},
            ("u", "plane", "u"):
                {"edge_index":u_plane_u},
            ("u", "nexus", "sp"):
                {"edge_index":u_nexus_sp},
            ("v", "plane", "v"):
                {"edge_index":v_plane_v},
            ("v", "nexus", "sp"):
                {"edge_index":v_nexus_sp},
            ("y", "plane", "y"):
                {"edge_index":y_plane_y},
            ("y", "nexus", "sp"):
                {"edge_index":y_nexus_sp} 

        }
    
        # Create HeteroData object & HeteroBatch
        hetero_data = HeteroData(output_dict)

        hetero_batch = Batch.from_data_list([hetero_data])
        return hetero_batch

class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """

        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args["model_config"])

        # Get ouptput configuration
        e_evt_config = pb_utils.get_output_config_by_name(model_config, "e_evt")
        x_semantic_u_config = pb_utils.get_output_config_by_name(model_config, "x_semantic_u")
        x_semantic_v_config = pb_utils.get_output_config_by_name(model_config, "x_semantic_v")
        x_semantic_y_config = pb_utils.get_output_config_by_name(model_config, "x_semantic_y")
        x_filter_u_config = pb_utils.get_output_config_by_name(model_config, "x_filter_u")
        x_filter_v_config = pb_utils.get_output_config_by_name(model_config, "x_filter_v")
        x_filter_y_config = pb_utils.get_output_config_by_name(model_config, "x_filter_y")
        v_evt_config = pb_utils.get_output_config_by_name(model_config, "v_evt")

        # Convert Triton types to numpy types
        self.e_evt_dtype = pb_utils.triton_string_to_numpy(
            e_evt_config["data_type"]
        )
        self.x_semantic_u_dtype = pb_utils.triton_string_to_numpy(
            x_semantic_u_config["data_type"]
        )
        self.x_semantic_v_dtype = pb_utils.triton_string_to_numpy(
            x_semantic_v_config["data_type"]
        )
        self.x_semantic_y_dtype = pb_utils.triton_string_to_numpy(
            x_semantic_y_config["data_type"]
        )
        self.x_filter_u_dtype = pb_utils.triton_string_to_numpy(
            x_filter_u_config["data_type"]
        )
        self.x_filter_v_dtype = pb_utils.triton_string_to_numpy(
            x_filter_v_config["data_type"]
        )
        self.x_filter_y_dtype = pb_utils.triton_string_to_numpy(
            x_filter_y_config["data_type"]
        )
        self.v_evt_dtype = pb_utils.triton_string_to_numpy(
            v_evt_config["data_type"]
        )
        # Instantiate the PyTorch model
        self.NuGraph3_model = NuGraph3_model()

    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse.

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        e_evt_dtype = self.e_evt_dtype
        x_semantic_u_dtype = self.x_semantic_u_dtype
        x_semantic_v_dtype = self.x_semantic_v_dtype
        x_semantic_y_dtype = self.x_semantic_y_dtype
        x_filter_u_dtype = self.x_filter_u_dtype
        x_filter_v_dtype = self.x_filter_v_dtype
        x_filter_y_dtype = self.x_filter_y_dtype
        v_evt_dtype = self.v_evt_dtype


        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            # Get all inputs
            sp_num_nodes = pb_utils.get_input_tensor_by_name(request, "sp_num_nodes")
            evt_y = pb_utils.get_input_tensor_by_name(request, "evt_y")

            u_x_dict = pb_utils.get_input_tensor_by_name(request, "u_x_dict")
            u_pos = pb_utils.get_input_tensor_by_name(request, "u_pos")
            v_x_dict = pb_utils.get_input_tensor_by_name(request, "v_x_dict")
            v_pos = pb_utils.get_input_tensor_by_name(request, "v_pos")
            y_x_dict = pb_utils.get_input_tensor_by_name(request, "y_x_dict")
            y_pos = pb_utils.get_input_tensor_by_name(request, "y_pos")

            u_plane_u = pb_utils.get_input_tensor_by_name(request, "u_plane_u")
            u_nexus_sp = pb_utils.get_input_tensor_by_name(request, "u_nexus_sp")
            v_plane_v = pb_utils.get_input_tensor_by_name(request, "v_plane_v")
            v_nexus_sp = pb_utils.get_input_tensor_by_name(request, "v_nexus_sp")
            y_plane_y = pb_utils.get_input_tensor_by_name(request, "y_plane_y")
            y_nexus_sp = pb_utils.get_input_tensor_by_name(request, "y_nexus_sp")
            
            examples = pb_utils.get_input_tensor_by_name(request, "examples")
            print(examples.as_numpy())

            output0, output1, output2, output3, output4, output5, output6, output7 = \
                                        self.NuGraph3_model(sp_num_nodes.as_numpy(), u_x_dict.as_numpy(), u_pos.as_numpy(), v_x_dict.as_numpy(), \
                                        v_pos.as_numpy(), y_x_dict.as_numpy(), y_pos.as_numpy(), \
                                        evt_y.as_numpy(), u_plane_u.as_numpy(), u_nexus_sp.as_numpy(), v_plane_v.as_numpy(), \
                                        v_nexus_sp.as_numpy(), y_plane_y.as_numpy(), y_nexus_sp.as_numpy())

            # Create output tensors. You need pb_utils.Tensor
            # objects to create pb_utils.InferenceResponse.
            out_tensor_0 = pb_utils.Tensor("e_evt", output0.astype(e_evt_dtype))
            out_tensor_1 = pb_utils.Tensor("x_semantic_u", output1.astype(x_semantic_u_dtype))
            out_tensor_2 = pb_utils.Tensor("x_semantic_v", output2.astype(x_semantic_v_dtype))
            out_tensor_3 = pb_utils.Tensor("x_semantic_y", output3.astype(x_semantic_y_dtype))
            out_tensor_4 = pb_utils.Tensor("x_filter_u", output4.astype(x_filter_u_dtype))
            out_tensor_5 = pb_utils.Tensor("x_filter_v", output5.astype(x_filter_v_dtype))
            out_tensor_6 = pb_utils.Tensor("x_filter_y", output6.astype(x_filter_y_dtype))
            out_tensor_7 = pb_utils.Tensor("v_evt", output7.astype(v_evt_dtype))
            # Create InferenceResponse. You can set an error here in case
            # there was a problem with handling this inference request.
            # Below is an example of how you can set errors in inference
            # response:
            #
            # pb_utils.InferenceResponse(
            #    output_tensors=..., TritonError("An error occurred"))
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0, out_tensor_1, out_tensor_2, out_tensor_3, out_tensor_4, out_tensor_5, out_tensor_6, out_tensor_7]
            )
            responses.append(inference_response)

        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print("Cleaning up...")