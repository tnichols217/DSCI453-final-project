"""
This type stub file was generated by pyright.
"""

from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export

"""xla is an experimental library that provides XLA support APIs."""
_XLA_COMPILE_ATTR = ...
_MAX_WARNING_LINES = ...
_DENYLISTED_OPS = ...
_UNSUPPORTED_OPS = ...
@tf_export('xla.experimental.compile')
@deprecated(None, 'xla.experimental.compile is deprecated. Consider using ' '`@tf.function(jit_compile=True)`.', warn_once=True)
def compile(computation, inputs=...): # -> object | _dispatcher_for_no_op | Operation | defaultdict[Any, Any] | list[Any] | list[Any | defaultdict[Any, Any] | list[Any] | object | None] | None:
  """Builds an operator that compiles and runs `computation` with XLA.

  NOTE: In eager mode, `computation` will have `@tf.function` semantics.

  Args:
    computation: A Python function that builds a computation to apply to the
      input. If the function takes n inputs, 'inputs' should be a list of n
      `Tensor`s.

      `computation` may return a list of `Tensor`s and `Operation`s.
      `Tensor`s must come before `Operation`s in the returned list.

      All `Operation`s returned from `computation` will be executed when
      evaluating any of the returned output tensors.
    inputs: A list of inputs or `None` (equivalent to an empty list). Each input
      can be a nested structure containing values that can be converted to
      `Tensor`s. Note that passing an N-dimension list of compatible values will
      result in an N-dimension list of scalar `Tensor`s rather than a single
      Rank-N `Tensor`. If you need a different behavior, convert parts of
      `inputs` to `Tensor`s with `tf.convert_to_tensor`.

  Returns:
    List of `Tensor`s corresponding to the `Tensor`s from
      the output of `computation` i.e. the same return value as if
      computation(*inputs) is called directly, with the following exceptions:
      * None output: a NoOp would be returned with a control dependency on
         `computation`.
      * Single value output: a tuple containing the value would be returned.
      * Operation-only outputs: a NoOp would be returned with a control
      dependency on `computation`.
      TODO(b/121383831): Investigate into removing these special cases.

  Raises:
    RuntimeError: When eager execution is enabled.

  Known issues:
    When a tf.random operation is built with XLA, the implementation doesn't
      pass the user provided seed to the XLA compiler. As such, the XLA compiler
      generates a random number and uses it as a seed when compiling the
      operation. This implementation causes a violation of the Tensorflow
      defined semantics in two aspects. First, changing the value of the user
      defined seed doesn't change the numbers generated by the operation.
      Second, when a seed is not specified, running the program multiple times
      will generate the same numbers.
  """
  ...

class XLACompileContext(control_flow_ops.XLAControlFlowContext):
  """A `ControlFlowContext` for nodes inside an XLA computation cluster.

  THIS IS ONLY FOR TENSORFLOW INTERNAL IMPLEMENTATION, DO NO USE DIRECTLY.

  The primary role of `XLACompileContext` is to mark operators inside a
  xla.compile() computation with attribute "_xla_compile_id=XYZ", where XYZ is
  a unique name.

  `ControlFlowContext` is used to perform the annotation since it integrates
  with Tensorflow constructs like ResourceVariables. For example, if a
  `ResourceVariable` is constructed inside a xla.compile() block, the
  `ResourceVariable` implementation can use
  `with ops.control_dependencies(None)` to build the variable's definition
  outside the compiled computation.
  """
  def __init__(self, name, pivot) -> None:
    """Builds a new XLACompileContext.

    Args:
      name: a unique name for the context, used to populate the
        `_xla_compile_id` attribute.
      pivot: a pivot node. Nodes in the XLACompileContext that do not have any
        inputs will have a control dependency on the pivot node. This ensures
        that nodes are correctly included in any enclosing control flow
        contexts.
    """
    ...
  
  def report_unsupported_operations(self): # -> None:
    ...
  
  def AddOp(self, op: ops.Operation): # -> None:
    """Create op in XLACompileContext and notifies outer context recursively."""
    ...
  
  def AddValue(self, val):
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
  


def is_flat(outputs): # -> bool:
  """Checks if outputs is a flat structure.

    Following structures and values are considered flat:
    1) None
    2) A single object
    3) A list or tuple of Tensors/Operations

    The only structures that this function understands are sequences,
    dictionaries and types defined using the attrs library.  E.g. this means
    that if outputs contains a single user-defined Object, it is considered to
    be flat. Errors are raised later on if that Object cannot be converted to a
    Tensor.

  Args:
    outputs: Output from `computation` inside `xla.compile`.

  Returns:
    A boolean indicates whether outputs is flat.
  """
  ...

class _CapturedObject:
  """A placeholder to capture an object."""
  def __init__(self) -> None:
    ...
  
  def capture(self, o): # -> None:
    ...
  
  def get(self): # -> None:
    ...
  


def check_function_argument_count(func, input_arity, infeed_queue): # -> LiteralString | None:
  """Validate the number of input arguments to an XLA function.

  Args:
    func: the Python function that will be called to generate the body of an XLA
      computation graph.
    input_arity: the number of explicit arguments supplied by the caller.
    infeed_queue: if not None, the infeed queue that will supply
      additional arguments to the function.

  Returns:
    None if function can be called with the supplied number of
      arguments, or an error string if it cannot.
  """
  ...

