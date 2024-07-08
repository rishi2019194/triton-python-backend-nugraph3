# Using Triton-Python-Backend for NuGraph3
In this repository, we discuss about how to deploy the NuGraph3 network at Fermilab using triton's python-backend interface. 
We provide the necessary libraries required for the setup and the commands to run the client and server side for running inference.

# Overview of the Setup
**Server Side:** In the server side, we recieve the data from the client for the - _particle_table_ and _spacepoint_table_. We then carry out the pre-processing necessary for creating a HeteroData Object as input for the NuGraph3 network. We then pass the pre-processed HeteroData Graph object through the saved checkpoint NuGraph3 model which we already have loaded via pytorch-lightning. After getting the output from the model, we send the results back to the client.

**Client Side:** In the client side, we first convert the read the H5 file/tables and flatten them into the necessary dictionary format of lists/vectors that
triton expects as part of the configuration file (config.pbtxt). After doing so we create InferInput objects and create HTTP/GRPC connection between the triton-server hosted at the local machine/EAF-sever and the client end. After doing so we send these InferInput objects to the server for inference. Once the inference is complete we get back the results at the client and display/save them.

# Installation Setup
We provide with the installation steps using both pip and conda. It is preferred to use conda-environment, especially when working the eaf-server.

## Docker Setup
Step 1:
 Install Docker using the following line:
 
    https://docs.docker.com/get-docker/
 
Step 2: 
  Pulling and Running the triton server imagee for docker
  
    docker pull nvcr.io/nvidia/tritonserver:<xx.yy>-py3
    docker run --shm-size=1g --ulimit memlock=-1 -p 8000:8000 -p 8001:8001 -p 8002:8002 --ulimit stack=67108864 -v "$(pwd)":/mnt -ti nvcr.io/nvidia/tritonserver:<xx.yy>-py3

    where <xx.yy> is triton version, eg - 24.05
  
Step 3:
   Starting the triton server for sending inference request
     tritonserver --model-repository models_folder/

## Using pip command
  To run the triton-server, install the following libraries in the base environment of the docker container -

    pip install --no-deps -e ./nugraph
    pip install pytorch_lightning
    pip install pynuml
    pip install matplotlib pynvml seaborn


## Using conda command

# Client-side inference setup and commands
We setup the client-side in - Python, C++ and C++ with LarSoft framework. It is preferred to use the C++ framework as we eventually want to shift to the LarSoft framework which is C++ compatible. 

## Python client inference

## C++ client inference


# Changes made to the existing code (Hacks at the server end)

