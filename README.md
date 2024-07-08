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

Step3:
  Cloning the triton-python-backend-nugraph3 repo inside the docker container

     git clone https://github.com/rishi2019194/triton-python-backend-nugraph3.git
     
Step 4:
   Starting the triton server for sending inference request (models-folder should be accessible from within the container and should follow the triton-expected format. Also, first install all the libraries using pip/conda)
   
     tritonserver --model-repository triton-python-backend-nugraph3/gnn_models_check/
     

## Using pip command
  To run the triton-server, install the following libraries in the base environment of the docker container -

    git clone https://github.com/rishi2019194/nugraph.git
    pip install --no-deps -e ./nugraph/nugraph
    pip install pytorch_lightning
    pip install pynuml
    pip install matplotlib pynvml seaborn


## Using conda command


# Client-side inference setup and commands
We setup the client-side in - Python, C++ and C++ with LarSoft framework. It is preferred to use the C++ with LarSoft framework

## Python client inference
To send inference request from Python-client, we first read the  H5 data file and then send the processed dictionary input for inference and display the results. 

  Libraries needed to run the python client file -

    pip install tritonclient[all]
    pip install pandas
    pip install numpy
    
  To run the python-based client, run the following commands after cloning this repo in your working directory -

    cd python_backend_scripts/
    python client_gnn_nugraph3_raw_data.py

## C++ client inference

## C++ client with LarSoft framework 

# Changes made to the existing code (Hacks at the server end)
## HitGraphProducer Class in Pynuml
In the current release version of [HitGraphProducer class](https://github.com/nugraph/pynuml/blob/main/pynuml/process/hitgraph.py#L10) of pynuml repo, the constructor requires **_file:'pynuml.io.File'_**, which is not possible since pre-processing is happening at the server end and we don't have access to the h5 file there for that event. Hence, as an hack we had to make our [own HitGraphProducer class](https://github.com/rishi2019194/triton-python-backend-nugraph3/blob/main/gnn_models_check/nugraph3_new/1/model.py) which doesn't have h5 file as one of the constructor's arguement. Apart from that we add a [**_create_graph()_**](https://github.com/rishi2019194/triton-python-backend-nugraph3/blob/main/gnn_models_check/nugraph3_new/1/model.py#L48) in that class which pre-processes the input sent from the client for inference.

## EventLabels Class in Nugraph
In the current version of [EventLabels class](https://github.com/nugraph/nugraph/blob/main/nugraph/nugraph/util/event_labels.py#L16) in nugraph repo, we first need to check if [**_data["evt"]_** has attribute **_'y'_** or not](https://github.com/rishi2019194/nugraph/blob/main/nugraph/nugraph/util/event_labels.py#L16). This is because during inference we won't have access to the ground truth labels of the event. Hence, we have added an if-condition to first check if _**'y'**_ is present as an attribute inside of _**data['evt']**_ or not.

## Changes in Nugraph3.py
In the current verison of [Nugraph3.py](https://github.com/nugraph/nugraph/blob/main/nugraph/nugraph/models/nugraph3/nugraph3.py#L207) in nugraph repo, we are calculating loss after computing the inference results in the step() function. However, in inference we don't have acccess to the ground truth y-labels, hence we can't calculate loss. Thus, as an hack we have c[ommented the loss computation step](https://github.com/rishi2019194/nugraph/blob/main/nugraph/nugraph/models/nugraph3/nugraph3.py#L208) in our code. Furthermore, to easily access the updated HeteroData object(data) which stores the inference result - we create a class attribute termed [**_data_**](https://github.com/rishi2019194/nugraph/blob/main/nugraph/nugraph/models/nugraph3/nugraph3.py#L205) via which we can access the inference  result and send that to  the  client.




