# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: brightness_service.proto
# Protobuf Python Version: 5.27.2
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC,
    5,
    27,
    2,
    '',
    'brightness_service.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x18\x62rightness_service.proto\x12\nbrightness\"\'\n\x0b\x43oordinates\x12\x0b\n\x03lat\x18\x01 \x01(\x01\x12\x0b\n\x03lon\x18\x02 \x01(\x01\"7\n\tPollution\x12\t\n\x01r\x18\x01 \x01(\r\x12\t\n\x01g\x18\x02 \x01(\r\x12\t\n\x01\x62\x18\x03 \x01(\r\x12\t\n\x01\x61\x18\x04 \x01(\r\"_\n\x15\x42rightnessObservation\x12\x0c\n\x04uuid\x18\x01 \x01(\t\x12\x0b\n\x03lat\x18\x02 \x01(\x01\x12\x0b\n\x03lon\x18\x03 \x01(\x01\x12\x0f\n\x07utc_iso\x18\x04 \x01(\t\x12\r\n\x05mpsas\x18\x05 \x01(\x02\x32\xaf\x01\n\x11\x42rightnessService\x12X\n\x18GetBrightnessObservation\x12\x17.brightness.Coordinates\x1a!.brightness.BrightnessObservation\"\x00\x12@\n\x0cGetPollution\x12\x17.brightness.Coordinates\x1a\x15.brightness.Pollution\"\x00\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'brightness_service_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_COORDINATES']._serialized_start=40
  _globals['_COORDINATES']._serialized_end=79
  _globals['_POLLUTION']._serialized_start=81
  _globals['_POLLUTION']._serialized_end=136
  _globals['_BRIGHTNESSOBSERVATION']._serialized_start=138
  _globals['_BRIGHTNESSOBSERVATION']._serialized_end=233
  _globals['_BRIGHTNESSSERVICE']._serialized_start=236
  _globals['_BRIGHTNESSSERVICE']._serialized_end=411
# @@protoc_insertion_point(module_scope)
