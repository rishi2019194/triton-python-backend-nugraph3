// Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <getopt.h>
#include <unistd.h>

#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <sstream>

#include "grpc_client.h"

namespace tc = triton::client;

#define FAIL_IF_ERR(X, MSG)                                        \
  {                                                                \
    tc::Error err = (X);                                           \
    if (!err.IsOk()) {                                             \
      std::cerr << "error: " << (MSG) << ": " << err << std::endl; \
      exit(1);                                                     \
    }                                                              \
  }

namespace {

void
Usage(char** argv, const std::string& msg = std::string())
{
  if (!msg.empty()) {
    std::cerr << "error: " << msg << std::endl;
  }

  std::cerr << "Usage: " << argv[0] << " [options]" << std::endl;
  std::cerr << "\t-v" << std::endl;
  std::cerr << "\t-m <model name>" << std::endl;
  std::cerr << "\t-u <URL for inference service>" << std::endl;
  std::cerr << "\t-t <client timeout in microseconds>" << std::endl;
  std::cerr << "\t-H <HTTP header>" << std::endl;
  std::cerr
      << "\tFor -H, header must be 'Header:Value'. May be given multiple times."
      << std::endl;
  std::cerr << "\t-C <grpc compression algorithm>. \'deflate\', "
               "\'gzip\' and \'none\' are supported"
            << std::endl;
  std::cerr << "\t-c <use_cached_channel>. "
               " Use cached channel when creating new client. "
               " Specify 'true' or 'false'. True by default"
            << std::endl;
  std::cerr << std::endl;

  exit(1);
}

}  // namespace

// Function to convert string to integer
int stoi(const std::string &str) {
    std::istringstream iss(str);
    int num;
    iss >> num;
    return num;
}


void printFloatArray(const float* data, size_t num_elements) {
    for (size_t i = 0; i < num_elements; ++i) {
        std::cout << data[i];
        if (i < num_elements - 1) {
            std::cout << " ";
        }
    }
    std::cout << std::endl;
}

// void triton_inference()

int
main(int argc, char** argv)
{
  bool verbose = false;
  std::string url("triton.fnal.gov:443");
  tc::Headers http_headers;
  uint32_t client_timeout = 0;
  bool use_ssl = true;
  std::string root_certificates;
  std::string private_key;
  std::string certificate_chain;
  grpc_compression_algorithm compression_algorithm =
      grpc_compression_algorithm::GRPC_COMPRESS_NONE;
  bool test_use_cached_channel = false;
  bool use_cached_channel = true;

  // {name, has_arg, *flag, val}
  static struct option long_options[] = {
      {"ssl", 0, 0, 0},
      {"root-certificates", 1, 0, 1},
      {"private-key", 1, 0, 2},
      {"certificate-chain", 1, 0, 3}};

  // Parse commandline...
  int opt;
  while ((opt = getopt_long(argc, argv, "vu:t:H:C:c:", long_options, NULL)) !=
         -1) {
    switch (opt) {
      case 0:
        use_ssl = true;
        break;
      case 1:
        root_certificates = optarg;
        break;
      case 2:
        private_key = optarg;
        break;
      case 3:
        certificate_chain = optarg;
        break;
      case 'v':
        verbose = true;
        break;
      case 'u':
        url = optarg;
        break;
      case 't':
        client_timeout = std::stoi(optarg);
        break;
      case 'H': {
        std::string arg = optarg;
        std::string header = arg.substr(0, arg.find(":"));
        if (header.size() == arg.size() || header.empty()) {
          Usage(
              argv,
              "HTTP header specified incorrectly. Must be formmated as "
              "'Header:Value'");
        } else {
          http_headers[header] = arg.substr(header.size() + 1);
        }
        break;
      }
      case 'C': {
        std::string algorithm_str{optarg};
        if (algorithm_str.compare("deflate") == 0) {
          compression_algorithm =
              grpc_compression_algorithm::GRPC_COMPRESS_DEFLATE;
        } else if (algorithm_str.compare("gzip") == 0) {
          compression_algorithm =
              grpc_compression_algorithm::GRPC_COMPRESS_GZIP;
        } else if (algorithm_str.compare("none") == 0) {
          compression_algorithm =
              grpc_compression_algorithm::GRPC_COMPRESS_NONE;
        } else {
          Usage(
              argv,
              "unsupported compression algorithm specified... only "
              "\'deflate\', "
              "\'gzip\' and \'none\' are supported.");
        }
        break;
      }
      case 'c': {
        test_use_cached_channel = true;
        std::string arg = optarg;
        if (arg.find("false") != std::string::npos) {
          use_cached_channel = false;
        } else if (arg.find("true") != std::string::npos) {
          use_cached_channel = true;
        } else {
          Usage(argv, "need to specify true or false for use_cached_channel");
        }
        break;
      }
      case '?':
        Usage(argv);
        break;
    }
  }

  // We use a simple model that takes 2 input tensors of 16 integers
  // each and returns 2 output tensors of 16 integers each. One output
  // tensor is the element-wise sum of the inputs and one output is
  // the element-wise difference.
  std::string model_name = "nugraph3";
  std::string model_version = "";

  // Create a InferenceServerGrpcClient instance to communicate with the
  // server using gRPC protocol.
  std::unique_ptr<tc::InferenceServerGrpcClient> client;
  tc::SslOptions ssl_options = tc::SslOptions();
  std::string err;
  if (use_ssl) {
    ssl_options.root_certificates = root_certificates;
    ssl_options.private_key = private_key;
    ssl_options.certificate_chain = certificate_chain;
    err = "unable to create secure grpc client";
  } else {
    err = "unable to create grpc client";
  }
  // Run with the same name to ensure cached channel is not used
  int numRuns = test_use_cached_channel ? 2 : 1;
  for (int i = 0; i < numRuns; ++i) {
    FAIL_IF_ERR(
        tc::InferenceServerGrpcClient::Create(
            &client, url, verbose, use_ssl, ssl_options, tc::KeepAliveOptions(),
            use_cached_channel),
        err);

    std::vector<int32_t> hit_table_hit_id_data;
    std::vector<int32_t> hit_table_local_plane_data;
    std::vector<float> hit_table_local_time_data;
    std::vector<int32_t> hit_table_local_wire_data;
    std::vector<float> hit_table_integral_data;
    std::vector<float> hit_table_rms_data;
    std::vector<int32_t> spacepoint_table_spacepoint_id_data;
    std::vector<int32_t> spacepoint_table_hit_id_u_data;
    std::vector<int32_t> spacepoint_table_hit_id_v_data;
    std::vector<int32_t> spacepoint_table_hit_id_y_data;

    std::ifstream infile("data.txt");
    if (!infile.is_open()) {
        std::cerr << "Error opening file: " << "data.txt" << std::endl;
    }

    std::string line;
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        std::string key;
        iss >> key;

        if (key == "hit_table_hit_id") {
            int num_elements;
            iss >> num_elements;
            hit_table_hit_id_data.resize(num_elements);
            for (int i = 0; i < num_elements; ++i) {
                iss >> hit_table_hit_id_data[i];
            }
        } else if (key == "hit_table_local_plane") {
            int num_elements;
            iss >> num_elements;
            hit_table_local_plane_data.resize(num_elements);
            for (int i = 0; i < num_elements; ++i) {
                iss >> hit_table_local_plane_data[i];
            }
        } else if (key == "hit_table_local_time") {
            int num_elements;
            iss >> num_elements;
            hit_table_local_time_data.resize(num_elements);
            for (int i = 0; i < num_elements; ++i) {
                iss >> hit_table_local_time_data[i];
            }
        } else if (key == "hit_table_local_wire") {
            int num_elements;
            iss >> num_elements;
            hit_table_local_wire_data.resize(num_elements);
            for (int i = 0; i < num_elements; ++i) {
                iss >> hit_table_local_wire_data[i];
            }
        } else if (key == "hit_table_integral") {
            int num_elements;
            iss >> num_elements;
            hit_table_integral_data.resize(num_elements);
            for (int i = 0; i < num_elements; ++i) {
                iss >> hit_table_integral_data[i];
            }
        } else if (key == "hit_table_rms") {
            int num_elements;
            iss >> num_elements;
            hit_table_rms_data.resize(num_elements);
            for (int i = 0; i < num_elements; ++i) {
                iss >> hit_table_rms_data[i];
            }
        } else if (key == "spacepoint_table_spacepoint_id") {
            int num_elements;
            iss >> num_elements;
            spacepoint_table_spacepoint_id_data.resize(num_elements);
            for (int i = 0; i < num_elements; ++i) {
                iss >> spacepoint_table_spacepoint_id_data[i];
            }
        } else if (key == "spacepoint_table_hit_id_u") {
            int num_elements;
            iss >> num_elements;
            spacepoint_table_hit_id_u_data.resize(num_elements);
            for (int i = 0; i < num_elements; ++i) {
                iss >> spacepoint_table_hit_id_u_data[i];
            }
        } else if (key == "spacepoint_table_hit_id_v") {
            int num_elements;
            iss >> num_elements;
            spacepoint_table_hit_id_v_data.resize(num_elements);
            for (int i = 0; i < num_elements; ++i) {
                iss >> spacepoint_table_hit_id_v_data[i];
            }
        } else if (key == "spacepoint_table_hit_id_y") {
            int num_elements;
            iss >> num_elements;
            spacepoint_table_hit_id_y_data.resize(num_elements);
            for (int i = 0; i < num_elements; ++i) {
                iss >> spacepoint_table_hit_id_y_data[i];
            }
        }
    }

    infile.close();


    std::vector<int64_t> hit_table_shape{int64_t(hit_table_hit_id_data.size())};
    std::vector<int64_t> spacepoint_table_shape{int64_t(spacepoint_table_spacepoint_id_data.size())};
  
    // Initialize the inputs with the data.
    tc::InferInput* hit_table_hit_id;
    tc::InferInput* hit_table_local_plane;
    tc::InferInput* hit_table_local_time;
    tc::InferInput* hit_table_local_wire;
    tc::InferInput* hit_table_integral;
    tc::InferInput* hit_table_rms;

    tc::InferInput* spacepoint_table_spacepoint_id;
    tc::InferInput* spacepoint_table_hit_id_u;
    tc::InferInput* spacepoint_table_hit_id_v;
    tc::InferInput* spacepoint_table_hit_id_y;


    FAIL_IF_ERR(
        tc::InferInput::Create(&hit_table_hit_id, "hit_table_hit_id", hit_table_shape, "INT32"),
        "unable to get hit_table_hit_id");
    std::shared_ptr<tc::InferInput> hit_table_hit_id_ptr;
    hit_table_hit_id_ptr.reset(hit_table_hit_id);

    FAIL_IF_ERR(
        tc::InferInput::Create(&hit_table_local_plane, "hit_table_local_plane", hit_table_shape, "INT32"),
        "unable to get hit_table_local_plane");
    std::shared_ptr<tc::InferInput> hit_table_local_plane_ptr;
    hit_table_local_plane_ptr.reset(hit_table_local_plane);

    FAIL_IF_ERR(
        tc::InferInput::Create(&hit_table_local_time, "hit_table_local_time", hit_table_shape, "FP32"),
        "unable to get hit_table_local_time");
    std::shared_ptr<tc::InferInput> hit_table_local_time_ptr;
    hit_table_local_time_ptr.reset(hit_table_local_time);

    FAIL_IF_ERR(
        tc::InferInput::Create(&hit_table_local_wire, "hit_table_local_wire", hit_table_shape, "INT32"),
        "unable to get hit_table_local_wire");
    std::shared_ptr<tc::InferInput> hit_table_local_wire_ptr;
    hit_table_local_wire_ptr.reset(hit_table_local_wire);

    FAIL_IF_ERR(
        tc::InferInput::Create(&hit_table_integral, "hit_table_integral", hit_table_shape, "FP32"),
        "unable to get hit_table_integral");
    std::shared_ptr<tc::InferInput> hit_table_integral_ptr;
    hit_table_integral_ptr.reset(hit_table_integral);

    FAIL_IF_ERR(
        tc::InferInput::Create(&hit_table_rms, "hit_table_rms", hit_table_shape, "FP32"),
        "unable to get hit_table_rms");
    std::shared_ptr<tc::InferInput> hit_table_rms_ptr;
    hit_table_rms_ptr.reset(hit_table_rms);


    FAIL_IF_ERR(
        tc::InferInput::Create(&spacepoint_table_spacepoint_id, "spacepoint_table_spacepoint_id", spacepoint_table_shape, "INT32"),
        "unable to get spacepoint_table_spacepoint_id");
    std::shared_ptr<tc::InferInput> spacepoint_table_spacepoint_id_ptr;
    spacepoint_table_spacepoint_id_ptr.reset(spacepoint_table_spacepoint_id);

    FAIL_IF_ERR(
        tc::InferInput::Create(&spacepoint_table_hit_id_u, "spacepoint_table_hit_id_u", spacepoint_table_shape, "INT32"),
        "unable to get spacepoint_table_spacepoint_hit_id_u");
    std::shared_ptr<tc::InferInput> spacepoint_table_hit_id_u_ptr;
    spacepoint_table_hit_id_u_ptr.reset(spacepoint_table_hit_id_u);

    FAIL_IF_ERR(
        tc::InferInput::Create(&spacepoint_table_hit_id_v, "spacepoint_table_hit_id_v", spacepoint_table_shape, "INT32"),
        "unable to get spacepoint_table_spacepoint_hit_id_v");
    std::shared_ptr<tc::InferInput> spacepoint_table_hit_id_v_ptr;
    spacepoint_table_hit_id_v_ptr.reset(spacepoint_table_hit_id_v);

    FAIL_IF_ERR(
        tc::InferInput::Create(&spacepoint_table_hit_id_y, "spacepoint_table_hit_id_y", spacepoint_table_shape, "INT32"),
        "unable to get spacepoint_table_spacepoint_hit_id_y");
    std::shared_ptr<tc::InferInput> spacepoint_table_hit_id_y_ptr;
    spacepoint_table_hit_id_y_ptr.reset(spacepoint_table_hit_id_y);



    FAIL_IF_ERR(
        hit_table_hit_id_ptr->AppendRaw(
            reinterpret_cast<uint8_t*>(&hit_table_hit_id_data[0]),
            hit_table_hit_id_data.size() * sizeof(float)),
        "unable to set data for hit_table_hit_id");

    FAIL_IF_ERR(
        hit_table_local_plane_ptr->AppendRaw(
            reinterpret_cast<uint8_t*>(&hit_table_local_plane_data[0]),
            hit_table_local_plane_data.size() * sizeof(float)),
        "unable to set data for hit_table_local_plane");
    
    FAIL_IF_ERR(
        hit_table_local_time_ptr->AppendRaw(
            reinterpret_cast<uint8_t*>(&hit_table_local_time_data[0]),
            hit_table_local_time_data.size() * sizeof(float)),
        "unable to set data for hit_table_local_time");

    FAIL_IF_ERR(
        hit_table_local_wire_ptr->AppendRaw(
            reinterpret_cast<uint8_t*>(&hit_table_local_wire_data[0]),
            hit_table_local_wire_data.size() * sizeof(float)),
        "unable to set data for hit_table_local_wire");

    FAIL_IF_ERR(
        hit_table_integral_ptr->AppendRaw(
            reinterpret_cast<uint8_t*>(&hit_table_integral_data[0]),
            hit_table_integral_data.size() * sizeof(float)),
        "unable to set data for hit_table_integral");

    FAIL_IF_ERR(
        hit_table_rms_ptr->AppendRaw(
            reinterpret_cast<uint8_t*>(&hit_table_rms_data[0]),
            hit_table_rms_data.size() * sizeof(float)),
        "unable to set data for hit_table_rms");

    FAIL_IF_ERR(
        spacepoint_table_spacepoint_id_ptr->AppendRaw(
            reinterpret_cast<uint8_t*>(&spacepoint_table_spacepoint_id_data[0]),
            spacepoint_table_spacepoint_id_data.size() * sizeof(float)),
        "unable to set data for spacepoint_table_spacepoint_id");

    FAIL_IF_ERR(
        spacepoint_table_hit_id_u_ptr->AppendRaw(
            reinterpret_cast<uint8_t*>(&spacepoint_table_hit_id_u_data[0]),
            spacepoint_table_hit_id_u_data.size() * sizeof(float)),
        "unable to set data for spacepoint_table_hit_id_u");

    FAIL_IF_ERR(
        spacepoint_table_hit_id_v_ptr->AppendRaw(
            reinterpret_cast<uint8_t*>(&spacepoint_table_hit_id_v_data[0]),
            spacepoint_table_hit_id_v_data.size() * sizeof(float)),
        "unable to set data for spacepoint_table_hit_id_v");

    FAIL_IF_ERR(
        spacepoint_table_hit_id_y_ptr->AppendRaw(
            reinterpret_cast<uint8_t*>(&spacepoint_table_hit_id_y_data[0]),
            spacepoint_table_hit_id_y_data.size() * sizeof(float)),
        "unable to set data for spacepoint_table_hit_id_y");

    // Generate the outputs to be requested.
    tc::InferRequestedOutput* e_evt;
    tc::InferRequestedOutput* x_semantic_u;
    tc::InferRequestedOutput* x_semantic_v;
    tc::InferRequestedOutput* x_semantic_y;
    tc::InferRequestedOutput* x_filter_u;
    tc::InferRequestedOutput* x_filter_v;
    tc::InferRequestedOutput* x_filter_y;
    tc::InferRequestedOutput* v_evt;

    FAIL_IF_ERR(
        tc::InferRequestedOutput::Create(&e_evt, "e_evt"),
        "unable to get 'e_evt'");
    std::shared_ptr<tc::InferRequestedOutput> e_evt_ptr;
    e_evt_ptr.reset(e_evt);

    FAIL_IF_ERR(
        tc::InferRequestedOutput::Create(&x_semantic_u, "x_semantic_u"),
        "unable to get 'x_semantic_u'");
    std::shared_ptr<tc::InferRequestedOutput> x_semantic_u_ptr;
    x_semantic_u_ptr.reset(x_semantic_u);

    FAIL_IF_ERR(
        tc::InferRequestedOutput::Create(&x_semantic_v, "x_semantic_v"),
        "unable to get 'x_semantic_v'");
    std::shared_ptr<tc::InferRequestedOutput> x_semantic_v_ptr;
    x_semantic_v_ptr.reset(x_semantic_v);

    FAIL_IF_ERR(
        tc::InferRequestedOutput::Create(&x_semantic_y, "x_semantic_y"),
        "unable to get 'x_semantic_y'");
    std::shared_ptr<tc::InferRequestedOutput> x_semantic_y_ptr;
    x_semantic_y_ptr.reset(x_semantic_y);

    FAIL_IF_ERR(
        tc::InferRequestedOutput::Create(&x_filter_u, "x_filter_u"),
        "unable to get 'x_filter_u'");
    std::shared_ptr<tc::InferRequestedOutput> x_filter_u_ptr;
    x_filter_u_ptr.reset(x_filter_u);

    FAIL_IF_ERR(
        tc::InferRequestedOutput::Create(&x_filter_v, "x_filter_v"),
        "unable to get 'x_filter_v'");
    std::shared_ptr<tc::InferRequestedOutput> x_filter_v_ptr;
    x_filter_v_ptr.reset(x_filter_v);

    FAIL_IF_ERR(
        tc::InferRequestedOutput::Create(&x_filter_y, "x_filter_y"),
        "unable to get 'x_filter_y'");
    std::shared_ptr<tc::InferRequestedOutput> x_filter_y_ptr;
    x_filter_y_ptr.reset(x_filter_y);

    FAIL_IF_ERR(
        tc::InferRequestedOutput::Create(&v_evt, "v_evt"),
        "unable to get 'v_evt'");
    std::shared_ptr<tc::InferRequestedOutput> v_evt_ptr;
    v_evt_ptr.reset(v_evt);

    // The inference settings. Will be using default for now.
    tc::InferOptions options(model_name);
    options.model_version_ = model_version;
    options.client_timeout_ = client_timeout;

    std::vector<tc::InferInput*> inputs = {hit_table_hit_id_ptr.get(), hit_table_local_plane_ptr.get(), hit_table_local_time_ptr.get(), \
                                          hit_table_local_wire_ptr.get(), hit_table_integral_ptr.get(), hit_table_rms_ptr.get(), \
                                          spacepoint_table_spacepoint_id_ptr.get(), spacepoint_table_hit_id_u_ptr.get(), \
                                          spacepoint_table_hit_id_v_ptr.get(), spacepoint_table_hit_id_y_ptr.get()};

    std::vector<const tc::InferRequestedOutput*> outputs = {
        e_evt_ptr.get(), x_semantic_u_ptr.get(), x_semantic_v_ptr.get(), \
        x_semantic_y_ptr.get(), x_filter_u_ptr.get(), x_filter_v_ptr.get(), \
        x_filter_y_ptr.get(), v_evt_ptr.get()};

    tc::InferResult* results;
    FAIL_IF_ERR(
        client->Infer(
            &results, options, inputs, outputs, http_headers,
            compression_algorithm),
        "unable to run model");
    std::shared_ptr<tc::InferResult> results_ptr;
    results_ptr.reset(results);

    // Get pointers to the result returned...

    const float* e_evt_data;
    size_t e_evt_byte_size;
    FAIL_IF_ERR(
        results_ptr->RawData(
            "e_evt", (const uint8_t**)&e_evt_data, &e_evt_byte_size),
        "unable to get result data for 'e_evt'");

    const float* x_semantic_u_data;
    size_t x_semantic_u_byte_size;
    FAIL_IF_ERR(
        results_ptr->RawData(
            "x_semantic_u", (const uint8_t**)&x_semantic_u_data, &x_semantic_u_byte_size),
        "unable to get result data for 'x_semantic_u'");

    const float* x_semantic_v_data;
    size_t x_semantic_v_byte_size;
    FAIL_IF_ERR(
        results_ptr->RawData(
            "x_semantic_v", (const uint8_t**)&x_semantic_v_data, &x_semantic_v_byte_size),
        "unable to get result data for 'x_semantic_v'");

    const float* x_semantic_y_data;
    size_t x_semantic_y_byte_size;
    FAIL_IF_ERR(
        results_ptr->RawData(
            "x_semantic_y", (const uint8_t**)&x_semantic_y_data, &x_semantic_y_byte_size),
        "unable to get result data for 'x_semantic_y'");

    const float* x_filter_u_data;
    size_t x_filter_u_byte_size;
    FAIL_IF_ERR(
        results_ptr->RawData(
            "x_filter_u", (const uint8_t**)&x_filter_u_data, &x_filter_u_byte_size),
        "unable to get result data for 'x_filter_u'");

    const float* x_filter_v_data;
    size_t x_filter_v_byte_size;
    FAIL_IF_ERR(
        results_ptr->RawData(
            "x_filter_v", (const uint8_t**)&x_filter_v_data, &x_filter_v_byte_size),
        "unable to get result data for 'x_filter_v'");

    const float* x_filter_y_data;
    size_t x_filter_y_byte_size;
    FAIL_IF_ERR(
        results_ptr->RawData(
            "x_filter_y", (const uint8_t**)&x_filter_y_data, &x_filter_y_byte_size),
        "unable to get result data for 'x_filter_y'");


    const float* v_evt_data;
    size_t v_evt_byte_size;
    FAIL_IF_ERR(
        results_ptr->RawData(
            "v_evt", (const uint8_t**)&v_evt_data, &v_evt_byte_size),
        "unable to get result data for 'v_evt'");

    std::cout<<"Trition output: "<<std::endl;

    std::cout<<"e_evt: "<<std::endl;
    printFloatArray(e_evt_data, e_evt_byte_size/sizeof(float));

    std::cout<<"x_semantic_u: "<<std::endl;
    printFloatArray(x_semantic_u_data, x_semantic_u_byte_size/sizeof(float));

    std::cout<<"x_semantic_v: "<<std::endl;
    printFloatArray(x_semantic_v_data, x_semantic_v_byte_size/sizeof(float));

    std::cout<<"x_semantic_y: "<<std::endl;
    printFloatArray(x_semantic_y_data, x_semantic_y_byte_size/sizeof(float));

    std::cout<<"x_filter_u: "<<std::endl;
    printFloatArray(x_filter_u_data, x_filter_u_byte_size/sizeof(float));

    std::cout<<"x_filter_v: "<<std::endl;
    printFloatArray(x_filter_v_data, x_filter_v_byte_size/sizeof(float));

    std::cout<<"x_filter_y: "<<std::endl;
    printFloatArray(x_filter_y_data, x_filter_y_byte_size/sizeof(float));

    std::cout<<"v_evt: "<<std::endl;
    printFloatArray(v_evt_data, v_evt_byte_size/sizeof(float));

  }
  return 0;
}
