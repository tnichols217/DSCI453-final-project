"""
This type stub file was generated by pyright.
"""

from tensorflow.security.fuzzing.py import annotation_types as _atypes
from tensorflow.python.util import dispatch as _dispatch
from tensorflow.python.util.tf_export import tf_export
from typing import Any, List, TypeVar
from typing_extensions import Annotated

"""Python wrappers around TensorFlow ops.

This file is MACHINE GENERATED! Do not edit.
"""
TV_XlaClusterOutput_T = TypeVar("TV_XlaClusterOutput_T", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_cluster_output')
def xla_cluster_output(input: Annotated[Any, TV_XlaClusterOutput_T], name=...) -> Annotated[Any, TV_XlaClusterOutput_T]:
  r"""Operator that connects the output of an XLA computation to other consumer graph nodes.

  Args:
    input: A `Tensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  ...

XlaClusterOutput = ...
_dispatcher_for_xla_cluster_output = xla_cluster_output._tf_type_based_dispatcher.Dispatch
def xla_cluster_output_eager_fallback(input: Annotated[Any, TV_XlaClusterOutput_T], name, ctx) -> Annotated[Any, TV_XlaClusterOutput_T]:
  ...

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_launch')
def xla_launch(constants, args, resources: Annotated[List[Any], _atypes.Resource], Tresults, function, name=...): # -> object | _dispatcher_for_xla_launch | Operation | tuple[Any, ...] | list[Any]:
  r"""XLA Launch Op. For use by the XLA JIT only.

  Args:
    constants: A list of `Tensor` objects.
    args: A list of `Tensor` objects.
    resources: A list of `Tensor` objects with type `resource`.
    Tresults: A list of `tf.DTypes`.
    function: A function decorated with @Defun.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `Tresults`.
  """
  ...

XlaLaunch = ...
_dispatcher_for_xla_launch = xla_launch._tf_type_based_dispatcher.Dispatch
def xla_launch_eager_fallback(constants, args, resources: Annotated[List[Any], _atypes.Resource], Tresults, function, name, ctx): # -> object:
  ...

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('xla_launch_v2')
def xla_launch_v2(args, Tresults, constants, resources, function, name=...): # -> object | _dispatcher_for_xla_launch_v2 | tuple[Any, ...] | list[Any]:
  r"""XLA Launch Op. For use by the XLA JIT only.

  Args:
    args: A list of `Tensor` objects.
    Tresults: A list of `tf.DTypes`.
    constants: A list of `ints`.
    resources: A list of `ints`.
    function: A function decorated with @Defun.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `Tresults`.
  """
  ...

XlaLaunchV2 = ...
_dispatcher_for_xla_launch_v2 = xla_launch_v2._tf_type_based_dispatcher.Dispatch
def xla_launch_v2_eager_fallback(args, Tresults, constants, resources, function, name, ctx): # -> object:
  ...

