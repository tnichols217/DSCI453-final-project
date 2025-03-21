"""
This type stub file was generated by pyright.
"""

from tensorflow.security.fuzzing.py import annotation_types as _atypes
from tensorflow.python.util import dispatch as _dispatch
from tensorflow.python.util.tf_export import tf_export
from typing import Any
from typing_extensions import Annotated

"""Python wrappers around TensorFlow ops.

This file is MACHINE GENERATED! Do not edit.
"""
@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('io.encode_proto')
def encode_proto(sizes: Annotated[Any, _atypes.Int32], values, field_names, message_type: str, descriptor_source: str = ..., name=...) -> Annotated[Any, _atypes.String]:
  r"""The op serializes protobuf messages provided in the input tensors.

  The types of the tensors in `values` must match the schema for the fields
  specified in `field_names`. All the tensors in `values` must have a common
  shape prefix, *batch_shape*.

  The `sizes` tensor specifies repeat counts for each field.  The repeat count
  (last dimension) of a each tensor in `values` must be greater than or equal
  to corresponding repeat count in `sizes`.

  A `message_type` name must be provided to give context for the field names.
  The actual message descriptor can be looked up either in the linked-in
  descriptor pool or a filename provided by the caller using the
  `descriptor_source` attribute.

  For the most part, the mapping between Proto field types and TensorFlow dtypes
  is straightforward. However, there are a few special cases:

  - A proto field that contains a submessage or group can only be converted
  to `DT_STRING` (the serialized submessage). This is to reduce the complexity
  of the API. The resulting string can be used as input to another instance of
  the decode_proto op.

  - TensorFlow lacks support for unsigned integers. The ops represent uint64
  types as a `DT_INT64` with the same twos-complement bit pattern (the obvious
  way). Unsigned int32 values can be represented exactly by specifying type
  `DT_INT64`, or using twos-complement if the caller specifies `DT_INT32` in
  the `output_types` attribute.

  The `descriptor_source` attribute selects the source of protocol
  descriptors to consult when looking up `message_type`. This may be:

  - An empty string  or "local://", in which case protocol descriptors are
  created for C++ (not Python) proto definitions linked to the binary.

  - A file, in which case protocol descriptors are created from the file,
  which is expected to contain a `FileDescriptorSet` serialized as a string.
  NOTE: You can build a `descriptor_source` file using the `--descriptor_set_out`
  and `--include_imports` options to the protocol compiler `protoc`.

  - A "bytes://<bytes>", in which protocol descriptors are created from `<bytes>`,
  which is expected to be a `FileDescriptorSet` serialized as a string.

  Args:
    sizes: A `Tensor` of type `int32`.
      Tensor of int32 with shape `[batch_shape, len(field_names)]`.
    values: A list of `Tensor` objects.
      List of tensors containing values for the corresponding field.
    field_names: A list of `strings`.
      List of strings containing proto field names.
    message_type: A `string`. Name of the proto message type to decode.
    descriptor_source: An optional `string`. Defaults to `"local://"`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  ...

EncodeProto = ...
_dispatcher_for_encode_proto = encode_proto._tf_type_based_dispatcher.Dispatch
def encode_proto_eager_fallback(sizes: Annotated[Any, _atypes.Int32], values, field_names, message_type: str, descriptor_source: str, name, ctx) -> Annotated[Any, _atypes.String]:
  ...

