"""
This type stub file was generated by pyright.
"""

from typing import Any, Callable, List, Text, Union
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops, variables
from tensorflow.python.types import core as core_types
from tensorflow.python.util.tf_export import tf_export

"""OutsideCompilation, TPUReplicateContext, and supporting functions."""
_MAX_WARNING_LINES = ...
_TPU_REPLICATE_ATTR = ...
_OUTSIDE_COMPILATION_ATTR = ...
_MAP_OUTSIDE_COMPILATION_ATTR = ...
_DENYLISTED_OPS = ...
_UNSUPPORTED_OPS = ...
def is_tpu_strategy(strategy: Any) -> bool:
  ...

class TPUReplicateContext(control_flow_ops.XLAControlFlowContext):
  """A `ControlFlowContext` for nodes inside a TPU computation.

  The primary role of `TPUReplicateContext` is to mark operators inside a
  tpu.replicate() computation with the attribute "_tpu_replicate=XYZ", where XYZ
  is a unique name.

  We use a `ControlFlowContext` to perform the annotation since it integrates
  with Tensorflow constructs like ResourceVariables. For example, if a
  `ResourceVariable` is constructed inside a tpu.replicate() block, the
  `ResourceVariable` implementation can use
  `with ops.control_dependencies(None)` to build the variable's definition
  outside the replicated computation.
  """
  def __init__(self, name: Text, num_replicas: int, pivot: ops.Operation) -> None:
    """Builds a new TPUReplicateContext.

    Args:
      name: a unique name for the context, used to populate the `_tpu_replicate`
        attribute.
      num_replicas: an integer that gives the number of replicas for the
        computation.
      pivot: a pivot node. Nodes in the TPUReplicateContext that do not have any
        inputs will have a control dependency on the pivot node. This ensures
        that nodes are correctly included in any enclosing control flow
        contexts.
    """
    ...
  
  def get_replicated_var_handle(self, name: Text, handle_id: Text, vars_: Union[List[core_types.Tensor], List[variables.Variable]], is_mirrored: bool = ..., is_packed: bool = ...) -> core_types.Tensor:
    """Returns a variable handle for replicated TPU variable 'var'.

    This is a method used by an experimental replicated variable implementation
    and is not intended as a public API.

    Args:
      name: The common name of the variable.
      handle_id: Unique ID of the variable handle, used as the cache key.
      vars_: The replicated TPU variables or handles.
      is_mirrored: Whether the variables are mirrored, which guarantees the
        values in each replica are always the same.
      is_packed: Whether the replicated variables are packed into one variable.

    Returns:
      The handle of the TPU replicated input node.
    """
    ...
  
  def report_unsupported_operations(self) -> None:
    ...
  
  def EnterGradientColocation(self, op: ops.Operation, gradient_uid: Text): # -> None:
    ...
  
  def ExitGradientColocation(self, op: ops.Operation, gradient_uid: Text): # -> None:
    ...
  
  def Enter(self) -> None:
    ...
  
  def HostComputeCore(self) -> List[Text]:
    ...
  
  def AddOp(self, op: ops.Operation) -> None:
    ...
  
  def AddValue(self, val: core_types.Tensor) -> core_types.Tensor:
    """Add `val` to the current context and its outer context recursively."""
    ...
  
  def AddInnerOp(self, op: ops.Operation): # -> None:
    ...
  
  @property
  def grad_state(self): # -> None:
    ...
  
  @property
  def back_prop(self): # -> Literal[False]:
    """Forwards to the enclosing while context, if any."""
    ...
  
  def GetControlPivot(self) -> ops.Operation:
    ...
  
  def RequiresUniqueFunctionRetracing(self): # -> Literal[True]:
    ...
  


class OutsideCompilationV2Context(control_flow_ops.ControlFlowContext):
  """The context for outside compilation in Tensorflow 2.0.

  Every op added in this context will be assigned an _xla_outside_compilation
  attribute.
  """
  def __init__(self, name: Text, is_map_outside_compilation=...) -> None:
    ...
  
  def AddOp(self, op: ops.Operation) -> None:
    ...
  
  def AddInnerOp(self, op: ops.Operation) -> None:
    ...
  
  def to_control_flow_context_def(self, context_def, export_scope=...):
    ...
  


def outside_compilation_impl(is_map, computation: Callable[..., Any], *args, **kwargs) -> Any:
  """Tags ops in `computation` with outside compilation attributes for ordinary `outside_compilation` or `map_outside_compilation`."""
  ...

@tf_export(v1=["tpu.outside_compilation"])
def outside_compilation(computation: Callable[..., Any], *args, **kwargs) -> Any:
  """Builds part of a computation outside any current TPU replicate scope.

  `tf.tpu.outside_compilation()` is used to run ops in `computation` on CPU
  instead of running on TPU. For example, users can run ops that are not
  supported on TPU's (e.g. tf.summary.write()) by explicitly placing those
  ops on CPU's. Below usage of outside compilation will place ops in
  `computation_with_string_ops` on CPU.

  Example usage:

  ```python
  def computation_with_string_ops(x):
    # strings types are not supported on TPU's and below ops must
    # run on CPU instead.
    output = tf.strings.format('1{}', x)
    return tf.strings.to_number(output)

  def tpu_computation():
    # Expected output is 11.
    output = tf.tpu.outside_compilation(computation_with_string_ops, 1)
  ```

  Outside compilation should be called inside TPUReplicateContext. That is,
  `tf.tpu.outside_compilation()` should be called inside a function that is
  passed to `tpu.split_compile_and_replicate()` -- this is implied when
  outside compilation is invoked inside a function passed to TPUStrategy
  `run()`. If invoked outside of TPUReplicateContext,
  then this simply returns the result of `computation`, and therefore,
  would be a no-op. Note that outside compilation is different from
  `tf.distribute.experimental.TPUStrategy.merge_call()` as logic in
  outside compilation is replicated and executed separately for each
  replica. On the other hand, `merge_call()` requires a `merge_fn`
  to aggregate the inputs from different replicas and is executed only
  once.

  For variables placed in TPU device, which includes variables created inside
  TPUStrategy scope, outside compilation logic must not include variable
  read/write. For variables placed on host, variable read/write is only allowed
  if the variable is not accessed by any other ops in the TPU computation.
  Variable read/write from outside compilation cluster is not visible from TPU
  computation and vice versa. Therefore, if outside compilation logic contains
  such host variables read/write ops and if the variables are accessed by TPU
  computation as well, then this may lead to deadlock.

  Internally, `tf.tpu.outside_compilation()` adds outside compilation
  attributes to all ops in `computation`. During a later passes ops with outside
  compilation attributes are moved to a host-side graph. Inputs to this extract
  host-side graph are sent from TPU computation graph to host graph via a pair
  of XlaSendToHost and XlaRecvFromHost ops. Note that using
  `tf.tpu.outside_compilation()` may result in tensor transfer between TPU and
  CPU, leading to non-trivial performance impact.

  Args:
    computation: A Python function that builds the computation to place on the
      host.
    *args: the positional arguments for the computation.
    **kwargs: the keyword arguments for the computation.

  Returns:
    The Tensors returned by computation.
  """
  ...

def experimental_map_outside_compilation(computation: Callable[..., Any], *args, **kwargs) -> Any:
  """Maps `computation` onto shards and puts it outside any current TPU replicate scope.

  `experimental_map_outside_compilation(f, x)` maps `f` onto the shards
  of `x`, where `x` is split-sharded. Each invocation of `f` on a split occurs
  on the CPU that's associated with the TPU that owns the split.

  Example usage:

  ```python
  def normalize_each_split(split):
    return split - tf.math.reduce_mean(split)

  def tpu_computation(x):
    x_split = strategy.experimental_split_to_logical_devices(
                x, [num_cores_per_replica, 1])
    y = experimental_map_outside_compilation(
          normalize_each_split, x_split)
    y_split = strategy.experimental_split_to_logical_devices(
                x, [num_cores_per_replica, 1])
    return y_split
  ```

  `experimental_map_outside_compilation` should be called inside
  TPUReplicateContext. That is, `outside_compilation()` should be called
  inside a function that is passed to `tpu.split_compile_and_replicate()` --
  this is implied when outside compilation is invoked inside a function passed
  to TPUStrategy `run()`. It is invalid to invoke outside of
  TPUReplicateContext.

  `experimental_map_outside_compilation` should input and output tensors that
  are located on the TPU.

  Internally, `experimental_map_outside_compilation()` adds outside
  compilation attributes to all ops in `computation` and moves outside-compiled
  ops to a host-side graph. This is similar to `tf.tpu.outside_compilation()`.
  Send/recv ops from/to the TPU send each split directly to the TPU's host.

  Args:
    computation: A Python function that builds the computation to place on the
      host.
    *args: the positional arguments for the computation.
    **kwargs: the keyword arguments for the computation.

  Returns:
    The Tensors returned by computation.
  """
  ...

