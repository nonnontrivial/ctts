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

aliased_import="import brightness_service_pb2 as brightness__service__pb2"
named_import="from . import brightness_service_pb2 as brightness__service__pb2"

sed -i '' -e "s/$aliased_import/$named_import/" ./pp/pp/stubs/brightness_service_pb2_grpc.py
sed -i '' -e "s/$aliased_import/$named_import/" ./api/api/stubs/brightness_service_pb2_grpc.py

echo "generated stubs"
