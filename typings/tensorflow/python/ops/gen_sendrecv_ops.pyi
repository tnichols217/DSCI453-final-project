"""
This type stub file was generated by pyright.
"""

from tensorflow.security.fuzzing.py import annotation_types as _atypes
from typing import Any, TypeVar
from typing_extensions import Annotated

"""Python wrappers around TensorFlow ops.

This file is MACHINE GENERATED! Do not edit.
"""
TV_Recv_tensor_type = TypeVar("TV_Recv_tensor_type", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
def recv(tensor_type: TV_Recv_tensor_type, tensor_name: str, send_device: str, send_device_incarnation: int, recv_device: str, client_terminated: bool = ..., name=...) -> Annotated[Any, TV_Recv_tensor_type]:
  r"""Receives the named tensor from send_device on recv_device.

  Args:
    tensor_type: A `tf.DType`.
    tensor_name: A `string`. The name of the tensor to receive.
    send_device: A `string`. The name of the device sending the tensor.
    send_device_incarnation: An `int`. The current incarnation of send_device.
    recv_device: A `string`. The name of the device receiving the tensor.
    client_terminated: An optional `bool`. Defaults to `False`.
      If set to true, this indicates that the node was added
      to the graph as a result of a client-side feed or fetch of Tensor data,
      in which case the corresponding send or recv is expected to be managed
      locally by the caller.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `tensor_type`.
  """
  ...

Recv = ...
def recv_eager_fallback(tensor_type: TV_Recv_tensor_type, tensor_name: str, send_device: str, send_device_incarnation: int, recv_device: str, client_terminated: bool, name, ctx) -> Annotated[Any, TV_Recv_tensor_type]:
  ...

TV_Send_T = TypeVar("TV_Send_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
def send(tensor: Annotated[Any, TV_Send_T], tensor_name: str, send_device: str, send_device_incarnation: int, recv_device: str, client_terminated: bool = ..., name=...): # -> object | Operation | None:
  r"""Sends the named tensor from send_device to recv_device.

  Args:
    tensor: A `Tensor`. The tensor to send.
    tensor_name: A `string`. The name of the tensor to send.
    send_device: A `string`. The name of the device sending the tensor.
    send_device_incarnation: An `int`. The current incarnation of send_device.
    recv_device: A `string`. The name of the device receiving the tensor.
    client_terminated: An optional `bool`. Defaults to `False`.
      If set to true, this indicates that the node was added
      to the graph as a result of a client-side feed or fetch of Tensor data,
      in which case the corresponding send or recv is expected to be managed
      locally by the caller.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  ...

Send = ...
def send_eager_fallback(tensor: Annotated[Any, TV_Send_T], tensor_name: str, send_device: str, send_device_incarnation: int, recv_device: str, client_terminated: bool, name, ctx): # -> None:
  ...

