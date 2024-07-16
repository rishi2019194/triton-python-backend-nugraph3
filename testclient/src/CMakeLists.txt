cmake_minimum_required (VERSION 3.12...3.14 FATAL_ERROR)

project(tritonclient_test VERSION 00.00.00)

find_package(cetbuildtools REQUIRED)

include(CetCMakeEnv)
cet_cmake_env()

cet_set_compiler_flags(DIAGS VIGILANT
  WERROR
  NO_UNDEFINED
  EXTRA_FLAGS -pedantic
  EXTRA_CXX_FLAGS -Wno-non-virtual-dtor -Wdelete-non-virtual-dtor)
cet_report_compiler_flags()

find_ups_product(protobuf)
find_ups_product(triton)
find_ups_product(grpc)

# For nugraph2
add_executable(simple_grpc_infer_client_nugraph2 simple_grpc_infer_client_nugraph2.cc)
target_include_directories(simple_grpc_infer_client_nugraph2 PRIVATE $ENV{GRPC_INC})
target_link_directories(
  simple_grpc_infer_client_nugraph2
  PRIVATE $ENV{TRITON_LIB}
          $ENV{GRPC_LIB}  
          $ENV{PROTOBUF_LIB})
target_link_libraries(simple_grpc_infer_client_nugraph2 protobuf grpcclient)

# For nugraph3
add_executable(simple_grpc_infer_client_nugraph3 simple_grpc_infer_client_nugraph3.cc)
target_include_directories(simple_grpc_infer_client_nugraph3 PRIVATE $ENV{GRPC_INC})
target_link_directories(
  simple_grpc_infer_client_nugraph3
  PRIVATE $ENV{TRITON_LIB}
          $ENV{GRPC_LIB}  
          $ENV{PROTOBUF_LIB})
target_link_libraries(simple_grpc_infer_client_nugraph3 protobuf grpcclient)