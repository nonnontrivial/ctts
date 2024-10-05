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

echo "done"

