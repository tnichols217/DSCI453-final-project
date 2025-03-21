"""
This type stub file was generated by pyright.
"""

import dataclasses
from typing import Any, Dict, List, Optional, Sequence, Union
from tensorflow.core.framework import attr_value_pb2, function_pb2, graph_debug_info_pb2
from tensorflow.core.function.polymorphism import function_type as function_type_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import func_graph as func_graph_module, ops
from tensorflow.python.types import core

"""Implementation for AtomicFunction."""
@dataclasses.dataclass(frozen=True)
class CallOptions:
  """Specifies additional configuration for an AtomicFunction call."""
  collective_manager_ids_used: List[int] = ...
  control_captures: List[Any] = ...
  is_stateful: bool = ...


RUNTIME_FUNCTION_REFS = ...
class AtomicFunction(core.AtomicFunction):
  """A Python callable for functions in the TF Runtime.

  Provides core functionality for tf.function including:
    - automatic lifecycle management of runtime functions
    - structured inputs (including captures) and structured outputs
    - calls from both eager and graph mode
    - dependency tracking of children functions
    - runtime error interpolation to identify user code stack traces
    - control dependencies (including automatic)
  """
  __slots__ = ...
  def __init__(self, name: Union[str, bytes], bound_context: context.Context, function_type: function_type_lib.FunctionType, children: Optional[List[AtomicFunction]] = ..., call_options: CallOptions = ..., cached_graph: Optional[func_graph_module.FuncGraph] = ...) -> None:
    """Construct a new AtomicFunction.

    Args:
      name: str/bytes name of the runtime function in the bound context.
      bound_context: interface to the runtime for the AtomicFunction.
      function_type: input/output contract for the AtomicFunction
      children: list of AtomicFunctions that are needed to call this one.
      call_options: extra configuration options for the call.
      cached_graph: FuncGraph that this AtomicFunction was generated from (if
        known). Otherwise it will lazily construct a new corresponding FuncGraph
        if ever needed.
    """
    ...
  
  @property
  def name(self) -> bytes:
    """Name represented in UTF-8 encoded bytes."""
    ...
  
  @property
  def function_type(self) -> function_type_lib.FunctionType:
    """Represents the input/output contract of this function."""
    ...
  
  @property
  def children(self) -> List[AtomicFunction]:
    """AtomicFunctions needed as dependencies for this one."""
    ...
  
  @property
  def definition(self) -> function_pb2.FunctionDef:
    """Current FunctionDef in the Runtime."""
    ...
  
  @property
  def attributes(self) -> Any:
    """Returns FunctionDef attributes in the Runtime."""
    ...
  
  @property
  def graph_debug_info(self) -> graph_debug_info_pb2.GraphDebugInfo:
    """A GraphDebugInfo proto mapping nodes to corresponding stack traces."""
    ...
  
  @property
  def call_options(self) -> CallOptions:
    """Call options declared for this AtomicFunction."""
    ...
  
  @property
  def graph_call_attrs(self) -> Dict[str, Any]:
    """Returns a dictionary of attributes needed to add a call in graph."""
    ...
  
  @property
  def cached_definition(self) -> function_pb2.FunctionDef:
    """Cached FunctionDef (not guaranteed to be fresh)."""
    ...
  
  @property
  def graph(self) -> func_graph_module.FuncGraph:
    """Returns a FuncGraph corresponding to the AtomicFunction."""
    ...
  
  def call_with_captures(self, args: Sequence[Any], kwargs: Dict[str, Any], captures: Sequence[Any]) -> Any:
    """Calls with args, kwargs, captures and returns structured output."""
    ...
  
  def call_preflattened(self, args: Sequence[core.Tensor]) -> Any:
    """Calls with flattened tensor inputs and returns the structured output."""
    ...
  
  def call_flat(self, *args: core.Tensor) -> Sequence[core.Tensor]:
    """Calls with flat tensor inputs and returns flat tensor outputs.

    Args:
      *args: arguments to call this function with.

    Returns:
      The outputs of the function call.

    Raises:
      ValueError: if the number of arguments is incorrect.
      FunctionAlreadyGarbageCollectedError: if the function is no longer
        available to be called because it has been garbage collected.
    """
    ...
  
  def __call__(self, *args, **kwargs) -> Any:
    ...
  
  def __del__(self): # -> None:
    ...
  
  def __str__(self) -> str:
    ...
  
  def __repr__(self): # -> str:
    ...
  


def partitioned_call_op(name: str, args: Sequence[core.Tensor], is_stateful: bool, tout: Sequence[Any], config: Any = ..., executor_type: Optional[str] = ..., xla_compile_attr: Any = ...) -> ops.Operation:
  """Generates a function call op respecting device annotations.

  Args:
    name: Name of the function to call.
    args: The arguments of the function, including captured inputs.
    is_stateful: If the function is stateful.
    tout: a list containing the output dtypes enums
    config: (Optional) A `tensorflow::ConfigProto` proto, serialized. If `None`,
      all optimizations are disabled. Currently only handled for eager defined
      functions.
    executor_type: (Optional) A string for the name of the executor to be used
      in the function call. If not set, or set to an empty string, the default
      tensorflow executor will be used.
    xla_compile_attr: (Optional) value of the XLA compilation attribute.

  Returns:
    Returns the operation.
  """
  ...

def make_call_op_in_graph(atomic: AtomicFunction, tensor_inputs: Sequence[core.Tensor], context_call_attrs: Dict[str, Any]): # -> list[Any]:
  """Adds an AtomicFunction to graph."""
  ...

def from_function_def(function_def: function_pb2.FunctionDef, function_type: function_type_lib.FunctionType) -> AtomicFunction:
  """Create a new AtomicFunction from FunctionDef + FunctionType."""
  ...

def from_func_graph(name: Union[str, bytes], graph: func_graph_module.FuncGraph, attrs: Dict[str, attr_value_pb2.AttrValue], function_type: Optional[function_type_lib.FunctionType] = ..., overwrite: bool = ...) -> AtomicFunction:
  """Initializes an AtomicFunction from FuncGraph.

  Args:
    name: str, the name for the created function.
    graph: Graph, the graph containing the operations in the function
    attrs: dict mapping names of attributes to their AttrValue values
    function_type: known FunctionType to use, otherwise one is derived.
    overwrite: overwrites function definition in the current context if needed

  Returns:
    An AtomicFunction instance.
  """
  ...

def to_func_graph(atomic: AtomicFunction) -> func_graph_module.FuncGraph:
  """Generate a FuncGraph from an AtomicFunction."""
  ...

class InterpolateRuntimeError:
  """Context Manager that interpolates exceptions received by AtomicFunction."""
  DENY_LIST_PHRASES = ...
  def __init__(self, top_level_func) -> None:
    ...
  
  def interpolate(self, message, node_names, graph_debug_info): # -> str:
    """Uses the GraphDebugInfo to generate an error message."""
    ...
  
  def __enter__(self): # -> None:
    ...
  
  def __exit__(self, typ, exc, tb): # -> Literal[False]:
    ...
  


