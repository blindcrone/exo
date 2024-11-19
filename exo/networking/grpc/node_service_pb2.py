# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: node_service.proto
# Protobuf Python Version: 5.28.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC,
    5,
    28,
    1,
    '',
    'node_service.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x12node_service.proto\x12\x0cnode_service\"S\n\x05Shard\x12\x10\n\x08model_id\x18\x01 \x01(\t\x12\x13\n\x0bstart_layer\x18\x02 \x01(\x05\x12\x11\n\tend_layer\x18\x03 \x01(\x05\x12\x10\n\x08n_layers\x18\x04 \x01(\x05\"k\n\rPromptRequest\x12\"\n\x05shard\x18\x01 \x01(\x0b\x32\x13.node_service.Shard\x12\x0e\n\x06prompt\x18\x02 \x01(\t\x12\x17\n\nrequest_id\x18\x03 \x01(\tH\x00\x88\x01\x01\x42\r\n\x0b_request_id\"\x81\x01\n\rTensorRequest\x12\"\n\x05shard\x18\x01 \x01(\x0b\x32\x13.node_service.Shard\x12$\n\x06tensor\x18\x02 \x01(\x0b\x32\x14.node_service.Tensor\x12\x17\n\nrequest_id\x18\x03 \x01(\tH\x00\x88\x01\x01\x42\r\n\x0b_request_id\"\xde\x01\n\x0e\x45xampleRequest\x12\"\n\x05shard\x18\x01 \x01(\x0b\x32\x13.node_service.Shard\x12%\n\x07\x65xample\x18\x02 \x01(\x0b\x32\x14.node_service.Tensor\x12$\n\x06target\x18\x03 \x01(\x0b\x32\x14.node_service.Tensor\x12$\n\x06length\x18\x04 \x01(\x0b\x32\x14.node_service.Tensor\x12\r\n\x05train\x18\x05 \x01(\x08\x12\x17\n\nrequest_id\x18\x06 \x01(\tH\x00\x88\x01\x01\x42\r\n\x0b_request_id\"V\n\x0bLossRequest\x12\"\n\x04loss\x18\x01 \x01(\x0b\x32\x14.node_service.Tensor\x12#\n\x05grads\x18\x02 \x01(\x0b\x32\x14.node_service.Tensor\"/\n\x19GetInferenceResultRequest\x12\x12\n\nrequest_id\x18\x01 \x01(\t\"\\\n\x0fInferenceResult\x12)\n\x06tensor\x18\x01 \x01(\x0b\x32\x14.node_service.TensorH\x00\x88\x01\x01\x12\x13\n\x0bis_finished\x18\x02 \x01(\x08\x42\t\n\x07_tensor\";\n\x06Tensor\x12\x13\n\x0btensor_data\x18\x01 \x01(\x0c\x12\r\n\x05shape\x18\x02 \x03(\x05\x12\r\n\x05\x64type\x18\x03 \x01(\t\"<\n\x16\x43ollectTopologyRequest\x12\x0f\n\x07visited\x18\x01 \x03(\t\x12\x11\n\tmax_depth\x18\x02 \x01(\x05\"\x8e\x02\n\x08Topology\x12\x30\n\x05nodes\x18\x01 \x03(\x0b\x32!.node_service.Topology.NodesEntry\x12\x39\n\npeer_graph\x18\x02 \x03(\x0b\x32%.node_service.Topology.PeerGraphEntry\x1aN\n\nNodesEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12/\n\x05value\x18\x02 \x01(\x0b\x32 .node_service.DeviceCapabilities:\x02\x38\x01\x1a\x45\n\x0ePeerGraphEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\"\n\x05value\x18\x02 \x01(\x0b\x32\x13.node_service.Peers:\x02\x38\x01\"\x19\n\x05Peers\x12\x10\n\x08peer_ids\x18\x01 \x03(\t\"7\n\x0b\x44\x65viceFlops\x12\x0c\n\x04\x66p32\x18\x01 \x01(\x02\x12\x0c\n\x04\x66p16\x18\x02 \x01(\x02\x12\x0c\n\x04int8\x18\x03 \x01(\x02\"k\n\x12\x44\x65viceCapabilities\x12\r\n\x05model\x18\x01 \x01(\t\x12\x0c\n\x04\x63hip\x18\x02 \x01(\t\x12\x0e\n\x06memory\x18\x03 \x01(\x05\x12(\n\x05\x66lops\x18\x04 \x01(\x0b\x32\x19.node_service.DeviceFlops\"L\n\x11SendResultRequest\x12\x12\n\nrequest_id\x18\x01 \x01(\t\x12\x0e\n\x06result\x18\x02 \x03(\x05\x12\x13\n\x0bis_finished\x18\x03 \x01(\x08\"=\n\x17SendOpaqueStatusRequest\x12\x12\n\nrequest_id\x18\x01 \x01(\t\x12\x0e\n\x06status\x18\x02 \x01(\t\"\x14\n\x12HealthCheckRequest\")\n\x13HealthCheckResponse\x12\x12\n\nis_healthy\x18\x01 \x01(\x08\"\x07\n\x05\x45mpty2\xf9\x04\n\x0bNodeService\x12\x41\n\nSendPrompt\x12\x1b.node_service.PromptRequest\x1a\x14.node_service.Tensor\"\x00\x12\x41\n\nSendTensor\x12\x1b.node_service.TensorRequest\x1a\x14.node_service.Tensor\"\x00\x12\x43\n\x0bSendExample\x12\x1c.node_service.ExampleRequest\x1a\x14.node_service.Tensor\"\x00\x12^\n\x12GetInferenceResult\x12\'.node_service.GetInferenceResultRequest\x1a\x1d.node_service.InferenceResult\"\x00\x12Q\n\x0f\x43ollectTopology\x12$.node_service.CollectTopologyRequest\x1a\x16.node_service.Topology\"\x00\x12\x44\n\nSendResult\x12\x1f.node_service.SendResultRequest\x1a\x13.node_service.Empty\"\x00\x12P\n\x10SendOpaqueStatus\x12%.node_service.SendOpaqueStatusRequest\x1a\x13.node_service.Empty\"\x00\x12T\n\x0bHealthCheck\x12 .node_service.HealthCheckRequest\x1a!.node_service.HealthCheckResponse\"\x00\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'node_service_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_TOPOLOGY_NODESENTRY']._loaded_options = None
  _globals['_TOPOLOGY_NODESENTRY']._serialized_options = b'8\001'
  _globals['_TOPOLOGY_PEERGRAPHENTRY']._loaded_options = None
  _globals['_TOPOLOGY_PEERGRAPHENTRY']._serialized_options = b'8\001'
  _globals['_SHARD']._serialized_start=36
  _globals['_SHARD']._serialized_end=119
  _globals['_PROMPTREQUEST']._serialized_start=121
  _globals['_PROMPTREQUEST']._serialized_end=228
  _globals['_TENSORREQUEST']._serialized_start=231
  _globals['_TENSORREQUEST']._serialized_end=360
  _globals['_EXAMPLEREQUEST']._serialized_start=363
  _globals['_EXAMPLEREQUEST']._serialized_end=585
  _globals['_LOSSREQUEST']._serialized_start=587
  _globals['_LOSSREQUEST']._serialized_end=673
  _globals['_GETINFERENCERESULTREQUEST']._serialized_start=675
  _globals['_GETINFERENCERESULTREQUEST']._serialized_end=722
  _globals['_INFERENCERESULT']._serialized_start=724
  _globals['_INFERENCERESULT']._serialized_end=816
  _globals['_TENSOR']._serialized_start=818
  _globals['_TENSOR']._serialized_end=877
  _globals['_COLLECTTOPOLOGYREQUEST']._serialized_start=879
  _globals['_COLLECTTOPOLOGYREQUEST']._serialized_end=939
  _globals['_TOPOLOGY']._serialized_start=942
  _globals['_TOPOLOGY']._serialized_end=1212
  _globals['_TOPOLOGY_NODESENTRY']._serialized_start=1063
  _globals['_TOPOLOGY_NODESENTRY']._serialized_end=1141
  _globals['_TOPOLOGY_PEERGRAPHENTRY']._serialized_start=1143
  _globals['_TOPOLOGY_PEERGRAPHENTRY']._serialized_end=1212
  _globals['_PEERS']._serialized_start=1214
  _globals['_PEERS']._serialized_end=1239
  _globals['_DEVICEFLOPS']._serialized_start=1241
  _globals['_DEVICEFLOPS']._serialized_end=1296
  _globals['_DEVICECAPABILITIES']._serialized_start=1298
  _globals['_DEVICECAPABILITIES']._serialized_end=1405
  _globals['_SENDRESULTREQUEST']._serialized_start=1407
  _globals['_SENDRESULTREQUEST']._serialized_end=1483
  _globals['_SENDOPAQUESTATUSREQUEST']._serialized_start=1485
  _globals['_SENDOPAQUESTATUSREQUEST']._serialized_end=1546
  _globals['_HEALTHCHECKREQUEST']._serialized_start=1548
  _globals['_HEALTHCHECKREQUEST']._serialized_end=1568
  _globals['_HEALTHCHECKRESPONSE']._serialized_start=1570
  _globals['_HEALTHCHECKRESPONSE']._serialized_end=1611
  _globals['_EMPTY']._serialized_start=1613
  _globals['_EMPTY']._serialized_end=1620
  _globals['_NODESERVICE']._serialized_start=1623
  _globals['_NODESERVICE']._serialized_end=2256
# @@protoc_insertion_point(module_scope)
