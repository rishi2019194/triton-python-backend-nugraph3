# Using Triton-Python-Backend for NuGraph3
In this repository, we discuss about how to deploy the NuGraph3 network at Fermilab using triton's python-backend interface. 
We provide the necessary libraries required for the setup and the commands to run the client and server side for running inference.

# Overview of the Setup
**Server Side:** In the server side, we recieve the data from the client for the - _particle_table_ and _spacepoint_table_. We then carry out the pre-processing 
necessary for creating a HeteroData Object as input for the NuGraph3 network. We then pass the pre-processed HeteroData Graph object through the saved checkpoint
NuGraph3 model which we already have loaded via pytorch-lightning. After getting the output from the model, we send the results back to the client.

**Client Side:** In the client side, we first convert the read the H5 file/tables and flatten them into the necessary dictionary format of lists/vectors that
triton expects as part of the configuration file (config.pbtxt). After doing so we create InferInput objects and create HTTP/GRPC connection between the triton-server
hosted at the local machine/EAF-sever and the client end. After doing so we send these InferInput objects to the server for inference. Once the inference is
complete we get back the results at the client and display/save them.

# Installation Setup
We provide with the installation steps using both pip and conda. It is preferred to use conda-environment, especially when working the eaf-server.

## Docker Setup

## Using pip command

## Using conda command

# Client-side inference setup and commands
We setup the client-side in both Python and C++. It is preferred to use the C++ framework as we eventually want to shift to the LarSoft framework which is 
C++ compatible. 

## Python client inference

## C++ client inference


# Changes made to the existing code (Hacks at the server end)

