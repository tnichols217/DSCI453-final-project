"""
This type stub file was generated by pyright.
"""

import abc
from tensorflow.python.framework import ops
from tensorflow.python.ops.gen_control_flow_ops import *
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export

"""Control Flow Operations.

See the [autograph](https://www.tensorflow.org/guide/autograph) guide.
"""
_basetuple = tuple
def exit(tensor, name=...): # -> Any | defaultdict[Any, Any] | list[Any] | object | None:
  """Exits the current frame to its parent frame.

  Exit makes its input `tensor` available to the parent frame.

  Args:
    tensor: The tensor to be made available to the parent frame.
    name: A name for this operation (optional).

  Returns:
    The same tensor as `tensor`.
  """
  ...

def switch(data, pred, dtype=..., name=...): # -> Switch | tuple[Any | defaultdict[Any, Any] | list[Any] | None, Any | defaultdict[Any, Any] | list[Any] | None]:
  """Forwards `data` to an output determined by `pred`.

  If `pred` is false, the `data` input is forwarded to the first output.
  Otherwise, the data goes to the second output.

  This op handles `Tensor`s and `IndexedSlices`.

  Args:
    data: The tensor to be forwarded to the appropriate output.
    pred: A scalar that specifies which output port will receive data.
    dtype: Optional element type for the returned tensor. If missing, the type
      is inferred from the type of `value`.
    name: A name for this operation (optional).

  Returns:
    `(output_false, output_true)`: If `pred` is true, data will be forwarded
    to `output_true`, otherwise it goes to `output_false`.
  """
  ...

def merge(inputs, name=...): # -> RefMerge | Merge | tuple[Any | defaultdict[Any, Any] | list[Any] | None, Any]:
  """Returns the value of an available element of `inputs`.

  This op tests each of the tensors in `inputs` in turn to determine if any of
  them is available. If it finds an available tensor, it returns it and its
  index in `inputs`.

  It is an error if more than one tensor in `inputs` is available. If no tensor
  in `inputs` is available, the returned tensor and index are not set.

  This op handles both `Tensor`s and `IndexedSlices`. If inputs has a mix of
  `Tensor`s and `IndexedSlices`, all inputs are converted to IndexedSlices
  before merging.

  Args:
    inputs: The input tensors, at most one of which is available.
    name: A name for this operation (optional).

  Returns:
    A tuple containing the chosen input tensor and its index in `inputs`.

  Raises:
    ValueError: If any of the inputs is None, or inputs are IndexedSlices and
      some but not all have a dense_shape property.
  """
  ...

class ControlFlowContext(metaclass=abc.ABCMeta):
  """The base class for control flow context.

  The usage pattern is a sequence of (Enter, Exit) followed by a final
  ExitResult.

  We maintain the following state for control flow contexts during graph
  construction:
   1. graph has _control_flow_context: the current context used to
      construct new nodes. Changed by ctxt.Enter() and ctxt.Exit()
   2. op has _control_flow_context: the context to which the op belongs.
      Set at the time the op is created. Immutable.
   3. A ControlFlowContext has _outer_context: the context in which this
      context is created. Set at the time a context is created. Immutable.
   4. A ControlFlowContext has _context_stack.
      Pushed and popped by ctxt.Enter() and ctxt.Exit()
  """
  def __init__(self, values_def=..., import_scope=...) -> None:
    ...
  
  @property
  def name(self):
    ...
  
  @property
  def outer_context(self): # -> None:
    """Return the context containing this context."""
    ...
  
  @property
  def grad_state(self):
    ...
  
  @property
  def back_prop(self):
    ...
  
  @abc.abstractmethod
  def to_control_flow_context_def(self, context_def, export_scope=...):
    """Serializes this into `context_def`.

    Args:
      context_def: a `ControlFlowContextDef` protocol buffer.
      export_scope: Optional `string`. Name scope to remove.
    """
    ...
  
  def AddName(self, name): # -> None:
    ...
  
  def Enter(self): # -> None:
    """Enter this control flow context."""
    ...
  
  def Exit(self): # -> None:
    """Exit this control flow context."""
    ...
  
  def EnterGradientColocation(self, op: ops.Operation, gradient_uid): # -> None:
    """Start building a gradient colocated with an op."""
    ...
  
  def ExitGradientColocation(self, op: ops.Operation, gradient_uid): # -> None:
    """Start building a gradient colocated with an op."""
    ...
  
  def ExitResult(self, result): # -> None:
    """Make a list of tensors available in the outer context."""
    ...
  
  def GetWhileContext(self): # -> None:
    """Return the while context containing this context."""
    ...
  
  def AddInnerOp(self, op: ops.Operation): # -> None:
    """Notifies a scope about an operator added to an inner scope."""
    ...
  
  def GetControlPivot(self): # -> None:
    """Returns the pivot node for this context, or None."""
    ...
  
  def IsWhileContext(self): # -> Literal[False]:
    ...
  
  def IsCondContext(self): # -> Literal[False]:
    ...
  
  def IsXLAContext(self): # -> Literal[False]:
    ...
  
  def __str__(self) -> str:
    ...
  


class CondContext(ControlFlowContext):
  """The context for the conditional construct."""
  def __init__(self, pred=..., pivot=..., branch=..., name=..., context_def=..., import_scope=...) -> None:
    """Creates a `CondContext`.

    Args:
      pred: The `boolean` tensor for the conditional predicate.
      pivot: The predicate tensor in this branch.
      branch: 0 or 1 representing this branch.
      name: Name of the `CondContext` python object.
      context_def: Optional `ContextDef` protocol buffer to initialize the
        `CondContext` object from.
      import_scope: Optional `string`. Name scope to add. Only used when
        initialing from protocol buffer.
    """
    ...
  
  @property
  def pred(self): # -> Tensor | Operation | None:
    ...
  
  @property
  def pivot(self): # -> Tensor | Operation | None:
    ...
  
  @property
  def branch(self): # -> None:
    ...
  
  @property
  def grad_state(self): # -> None:
    ...
  
  @property
  def back_prop(self): # -> Literal[False]:
    ...
  
  def GetControlPivot(self): # -> Tensor | Operation | None:
    ...
  
  def to_proto(self, export_scope=...): # -> None:
    """Converts a `CondContext` to a `CondContextDef` protocol buffer.

    Args:
      export_scope: Optional `string`. Name scope to remove.

    Returns:
      A `CondContextDef` protocol buffer.
    """
    ...
  
  @staticmethod
  def from_proto(context_def, import_scope=...): # -> CondContext:
    """Returns a `CondContext` object created from `context_def`."""
    ...
  
  def to_control_flow_context_def(self, context_def, export_scope=...): # -> None:
    ...
  
  def AddValue(self, val):
    """Add `val` to the current context and its outer context recursively."""
    ...
  
  def AddOp(self, op: ops.Operation): # -> None:
    ...
  
  def BuildCondBranch(self, fn): # -> tuple[object | _dispatcher_for_no_op | Any | Operation | None, None] | tuple[None, None] | tuple[Any | defaultdict[Any, Any] | list[Any] | object | None, list[Any | defaultdict[Any, Any] | object | None] | list[Any] | _basetuple]:
    """Add the subgraph defined by fn() to the graph."""
    ...
  
  def IsCondContext(self): # -> Literal[True]:
    ...
  


class WhileContext(ControlFlowContext):
  """The context for the loop construct."""
  def __init__(self, maximum_iterations=..., parallel_iterations=..., back_prop=..., swap_memory=..., name=..., grad_state=..., context_def=..., import_scope=...) -> None:
    """"Creates a `WhileContext`.

    Args:
      maximum_iterations: Optional upper bound on number of loop iterations.
      parallel_iterations: The number of iterations allowed to run in parallel.
      back_prop: Whether backprop is enabled for this while loop.
      swap_memory: Whether GPU-CPU memory swap is enabled for this loop.
      name: Optional name prefix for the returned tensors.
      grad_state: The gradient loop state.
      context_def: Optional `WhileContextDef` protocol buffer to initialize the
        `Whilecontext` python object from.
      import_scope: Optional `string`. Name scope to add. Only used when
        initialing from protocol buffer.
    """
    ...
  
  @property
  def maximum_iterations(self): # -> Tensor | Operation | None:
    """The maximum number of iterations that will be executed."""
    ...
  
  @property
  def parallel_iterations(self): # -> int:
    """The number of iterations allowed to run in parallel."""
    ...
  
  @property
  def back_prop(self):
    """True iff backprop is enabled for this while loop."""
    ...
  
  @property
  def swap_memory(self):
    """True iff GPU-CPU memory swap is enabled for this while loop."""
    ...
  
  @property
  def pivot(self): # -> Tensor | Operation | Any | None:
    """The boolean tensor representing the loop termination condition."""
    ...
  
  @property
  def loop_enters(self): # -> list[Tensor | Operation]:
    """The list of enter tensors for loop variables."""
    ...
  
  @property
  def loop_exits(self): # -> list[Tensor | Operation] | list[Any | defaultdict[Any, Any] | list[Any] | object | None]:
    """The list of exit tensors for loop variables."""
    ...
  
  @property
  def grad_state(self): # -> None:
    """The gradient loop state."""
    ...
  
  def to_proto(self, export_scope=...): # -> None:
    """Converts a `WhileContext` to a `WhileContextDef` protocol buffer.

    Args:
      export_scope: Optional `string`. Name scope to remove.

    Returns:
      A `WhileContextDef` protocol buffer.
    """
    ...
  
  def to_control_flow_context_def(self, context_def, export_scope=...): # -> None:
    ...
  
  @staticmethod
  def from_proto(context_def, import_scope=...): # -> WhileContext:
    """Returns a `WhileContext` object created from `context_def`.

    Args:
      context_def: A `WhileContextDef` protocol buffer.
      import_scope: Optional `string`. Name scope to add.

    Returns:
      A `WhileContext` Python object.
    """
    ...
  
  def GetWhileContext(self): # -> Self:
    ...
  
  def GetControlPivot(self): # -> Tensor | Operation | Any | defaultdict[Any, Any] | list[Any] | object | None:
    ...
  
  def AddValue(self, val): # -> Any | defaultdict[Any, Any] | list[Any] | object | None:
    """Add `val` to the current context and its outer context recursively."""
    ...
  
  def AddOp(self, op: ops.Operation): # -> None:
    """Add `op` to the current context."""
    ...
  
  def AddForwardLoopCounter(self, outer_grad_state): # -> tuple[Any | defaultdict[Any, Any] | list[Any] | object | None, Any | defaultdict[Any, Any] | list[Any] | object | None]:
    """Adds a loop that counts the number of iterations.

    This is added to the forward loop at the time when we start to
    create the loop for backprop gradient computation. Called in
    the outer context of this forward context.

    The pseudocode is:
      `n = 0; while (_pivot) { n++; }`

    Note that a control dependency is added to `n` to ensure the correct
    execution order of stack push ops.

    Args:
      outer_grad_state: The outer grad state. None if not nested.

    Returns:
      The number of iterations taken by the forward loop and the loop index.
    """
    ...
  
  def AddBackpropLoopCounter(self, count, outer_grad_state): # -> Any | defaultdict[Any, Any] | list[Any] | object | None:
    """Add the backprop loop that controls the iterations.

    This is added to the backprop loop. It is used to control the loop
    termination of the backprop loop. Called in the outer context of
    this grad context.

    The pseudocode is:
      `n = count; while (n >= 1) { n--; }`

    Note that a control dependency is added to `final_zero` to ensure the
    correct execution order of stack pop ops.

    Args:
      count: The number of iterations for backprop.
      outer_grad_state: The outer grad state. None if not nested.

    Returns:
      The loop index.
    """
    ...
  
  def AddBackpropAccumulator(self, op: ops.Operation, grad): # -> Any | defaultdict[Any, Any] | list[Any] | object | None:
    """Add an accumulation loop for every loop invariant.

    This is added to the backprop loop. It is used to accumulate partial
    gradients within each loop iteration. Called when in the gradient while
    context.

    The pseudocode is:
      ```
      acc = 0.0;
      while (_pivot) {
        acc += grad;
      }
      ```

    Args:
      op: The Enter op for a loop invariant.
      grad: The partial gradient of an iteration for a loop invariant.

    Returns:
      The gradient for a loop invariant.
    """
    ...
  
  def AddBackpropIndexedSlicesAccumulator(self, op: ops.Operation, grad): # -> IndexedSlices:
    """This is used for accumulating gradients that are IndexedSlices.

    This is essentially the equivalent of AddBackpropAccumulator but optimized
    for things like updating embeddings from within a while loop.

    Args:
      op: The Enter op for a loop invariant.
      grad: The partial gradients represented as an IndexedSlices.

    Returns:
      The accumulated IndexedSlices gradient of the loop invariant.
    """
    ...
  
  def BuildLoop(self, pred, body, loop_vars, shape_invariants, return_same_structure): # -> defaultdict[Any, Any] | Any | list[Any] | None:
    """Add the loop termination condition and body to the graph."""
    ...
  
  def IsWhileContext(self): # -> Literal[True]:
    ...
  


def with_dependencies(dependencies, output_tensor, name=...): # -> IndexedSlices | Any | defaultdict[Any, Any] | list[Any] | object | None:
  """Produces the content of `output_tensor` only after `dependencies`.

  In some cases, a user may want the output of an operation to be
  consumed externally only after some other dependencies have run
  first. This function ensures returns `output_tensor`, but only after all
  operations in `dependencies` have run. Note that this means that there is
  no guarantee that `output_tensor` will be evaluated after any `dependencies`
  have run.

  See also `tf.tuple` and `tf.group`.

  Args:
    dependencies: Iterable of operations to run before this op finishes.
    output_tensor: A `Tensor` or `IndexedSlices` that will be returned.
    name: (Optional) A name for this operation.

  Returns:
    Same as `output_tensor`.

  Raises:
    TypeError: if `output_tensor` is not a `Tensor` or `IndexedSlices`.
  """
  ...

@tf_export("group")
def group(*inputs, **kwargs): # -> object | _dispatcher_for_no_op | Operation | None:
  """Create an op that groups multiple operations.

  When this op finishes, all ops in `inputs` have finished. This op has no
  output.

  Note: *In TensorFlow 2 with eager and/or Autograph, you should not require
  this method, as ops execute in the expected order thanks to automatic control
  dependencies.* Only use `tf.group` when working with v1
  `tf.Graph` code.

  When operating in a v1-style graph context, ops are not executed in the same
  order as specified in the code; TensorFlow will attempt to execute ops in
  parallel or in an order convenient to the result it is computing.  `tf.group`
  allows you to request that one or more results finish before execution
  continues.

  `tf.group` creates a single op (of type `NoOp`), and then adds appropriate
  control dependencies.  Thus, `c = tf.group(a, b)` will compute the same graph
  as this:

      with tf.control_dependencies([a, b]):
          c = tf.no_op()

  See also `tf.tuple` and
  `tf.control_dependencies`.

  Args:
    *inputs: Zero or more tensors to group.
    name: A name for this operation (optional).

  Returns:
    An Operation that executes all its inputs.

  Raises:
    ValueError: If an unknown keyword argument is provided.
  """
  ...

@tf_export("tuple", v1=[])
@dispatch.add_dispatch_support
def tuple_v2(tensors, control_inputs=..., name=...): # -> list[Any]:
  """Groups tensors together.

  The returned tensors have the same value as the input tensors, but they
  are computed only after all the input tensors have been computed.

  Note: *In TensorFlow 2 with eager and/or Autograph, you should not require
  this method, as ops execute in the expected order thanks to automatic control
  dependencies.* Only use `tf.tuple` when working with v1 `tf.Graph` code.

  See also `tf.group` and `tf.control_dependencies`.

  Example:
  >>> with tf.Graph().as_default():
  ...   with tf.compat.v1.Session() as sess:
  ...     v = tf.Variable(0.0)
  ...     a = tf.constant(1.0)
  ...     sess.run(tf.compat.v1.global_variables_initializer())
  ...     for i in range(5):
  ...       update_op = v.assign_add(1.0)
  ...       b = a + v
  ...       res_b = sess.run(b)
  ...       res_v = sess.run(v)
  ...       print(res_v)
  0.0
  0.0
  0.0
  0.0
  0.0

  >>> with tf.Graph().as_default():
  ...   with tf.compat.v1.Session() as sess:
  ...     v = tf.Variable(0.0)
  ...     a = tf.constant(1.0)
  ...     sess.run(tf.compat.v1.global_variables_initializer())
  ...     for i in range(5):
  ...       update_op = v.assign_add(1.0)
  ...       calc = [a + v]
  ...       # `tf.tuple` ensures `update_op` is run before `b`
  ...       b = tf.tuple(calc, [tf.group(update_op)])
  ...       res_b = sess.run(b)
  ...       res_v = sess.run(v)
  ...       print(res_v)
  1.0
  2.0
  3.0
  4.0
  5.0


  Args:
    tensors: A list of `Tensor`s or `IndexedSlices`, some entries can be `None`.
    control_inputs: List of additional ops to finish before returning.
    name: (optional) A name to use as a `name_scope` for the operation.

  Returns:
    Same as `tensors`.

  Raises:
    ValueError: If `tensors` does not contain any `Tensor` or `IndexedSlices`.
    TypeError: If `control_inputs` is not a list of `Operation` or `Tensor`
      objects.

  """
  ...

@tf_export(v1=["tuple"])
@dispatch.add_dispatch_support
def tuple(tensors, name=..., control_inputs=...): # -> list[Any]:
  """Group tensors together.

  This creates a tuple of tensors with the same values as the `tensors`
  argument, except that the value of each tensor is only returned after the
  values of all tensors have been computed.

  `control_inputs` contains additional ops that have to finish before this op
  finishes, but whose outputs are not returned.

  This can be used as a "join" mechanism for parallel computations: all the
  argument tensors can be computed in parallel, but the values of any tensor
  returned by `tuple` are only available after all the parallel computations
  are done.

  See also `tf.group` and
  `tf.control_dependencies`.

  Args:
    tensors: A list of `Tensor`s or `IndexedSlices`, some entries can be `None`.
    name: (optional) A name to use as a `name_scope` for the operation.
    control_inputs: List of additional ops to finish before returning.

  Returns:
    Same as `tensors`.

  Raises:
    ValueError: If `tensors` does not contain any `Tensor` or `IndexedSlices`.
    TypeError: If `control_inputs` is not a list of `Operation` or `Tensor`
      objects.

  """
  ...

class XLAControlFlowContext(ControlFlowContext):
  """Base class for XLA and TPU control flow contexts."""
  def __init__(self) -> None:
    ...
  
  def to_control_flow_context_def(self, context_def, export_scope=...): # -> None:
    ...
  
  def IsXLAContext(self): # -> Literal[True]:
    ...
  
  def AddOp(self, _): # -> None:
    ...
  
  def AddValue(self, x):
    ...
  
  def RequiresUniqueFunctionRetracing(self): # -> Literal[False]:
    """Returns whether the tf.function should be retraced if the context changes.
    """
    ...
  


@tf_export("__internal__.get_enclosing_xla_context", v1=[])
def get_enclosing_xla_context(): # -> XLAControlFlowContext | None:
  """Recursively find and return the XLAControlFlowContext."""
  ...

def from_control_flow_context_def(context_def, import_scope=...): # -> CondContext | WhileContext:
  """Deserializes `context_def` into the appropriate ControlFlowContext.

  Args:
    context_def: ControlFlowContextDef proto
    import_scope: Optional `string`. Name scope to add.

  Returns:
    A ControlFlowContext subclass
  """
  ...

