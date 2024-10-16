#!/bin/bash

generate_stubs() {
  pathname="$1"
  python3 -m grpc_tools.protoc \
      --proto_path=protos \
      --python_out=$pathname \
      --grpc_python_out=$pathname \
      protos/*.proto
}


generate_stubs "./pp/pp/stubs"
generate_stubs "./api/api/stubs"

old="import brightness_service_pb2 as brightness__service__pb2"
new="from . import brightness_service_pb2 as brightness__service__pb2"

sed -i '' -e "s/$old/$new/" ./pp/pp/stubs/brightness_service_pb2_grpc.py
sed -i '' -e "s/$old/$new/" ./api/api/stubs/brightness_service_pb2_grpc.py

echo "generated stubs"

