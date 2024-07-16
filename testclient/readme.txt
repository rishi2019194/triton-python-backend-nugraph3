# set up the environment, replace the source /products/setup command with
# whatever you use to set up the UPS products 
source /products/setup
setup cetbuildtools v8_20_00
setup triton v2_25_0d -q e26
setup grpc v1_35_0c -q e26

# goto ./testclient subdir
cd build
source ../src/ups/setup_for_development -p
buildtool -c

# after building, your executable will be in:
# ./build/slf7.x86_64.e26.prof/bin/simple_grpc_infer_client
