# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "grpcio",
#     "grpcio-tools",
#     "protobuf",
# ]
# ///

import logging
from pathlib import Path

import grpc_tools.protoc

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def generate_stubs_at_path(output_path: Path, adjust_imports=False):
    import subprocess

    proto_path = list(Path.cwd().parent.parent.rglob("brightness_service.proto"))[0]
    grpc_tools.protoc.main(
        (
            "",
            f"--proto_path={proto_path.parent.as_posix()}",
            f"--python_out={output_path.as_posix()}",
            f"--grpc_python_out={output_path.as_posix()}",
            proto_path.as_posix(),
        )
    )
    if adjust_imports:
        # avoid ModuleNotFound errors by using sed to use named import
        aliased_import = "import brightness_service_pb2 as brightness__service__pb2"
        named_import = (
            "from . import brightness_service_pb2 as brightness__service__pb2"
        )
        pb2_paths = list(output_path.rglob("brightness_service_pb2_grpc.py"))
        for path in pb2_paths:
            try:
                cmd = [
                    "sed",
                    "-i",
                    "",
                    "-e",
                    f"s/{aliased_import}/{named_import}/",
                    path.as_posix(),
                ]
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                log.error(f"failed to edit stub {e}")


def main() -> None:
    """generate stubs according to the proto file"""
    service_stub_paths = list(Path.cwd().parent.rglob("stubs"))
    for path in service_stub_paths:
        generate_stubs_at_path(path, adjust_imports=True)

    script_stub_paths = Path.cwd() / "stubs"
    script_stub_paths.mkdir(exist_ok=True)
    generate_stubs_at_path(script_stub_paths, adjust_imports=False)
    (script_stub_paths / "__init__.py").touch(exist_ok=True)


if __name__ == "__main__":
    main()
