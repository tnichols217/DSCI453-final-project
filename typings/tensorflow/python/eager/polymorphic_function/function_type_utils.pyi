"""
This type stub file was generated by pyright.
"""

import inspect
from typing import Any, Dict, Tuple
from tensorflow.core.function import trace_type
from tensorflow.core.function.polymorphism import function_type as function_type_lib

"""Utilities for using FunctionType with tf.function."""
def to_fullargspec(function_type: function_type_lib.FunctionType, default_values: Dict[str, Any]) -> inspect.FullArgSpec:
  """Generates backwards compatible FullArgSpec from FunctionType."""
  ...

def to_function_type(fullargspec): # -> tuple[FunctionType, dict[str, Any]]:
  """Generates FunctionType and default values from fullargspec."""
  ...

def to_input_signature(function_type): # -> tuple[Any, ...] | None:
  """Extracts an input_signature from function_type instance."""
  ...

def to_arg_names(function_type): # -> list[Any]:
  """Generates a list of arg names from a FunctionType."""
  ...

class FunctionSpec:
  """Specification of how to bind arguments to a function.

  Deprecated. Please use FunctionType instead.
  """
  @classmethod
  def from_function_and_signature(cls, python_function, input_signature, is_pure=..., jit_compile=...): # -> FunctionSpec:
    """Creates a FunctionSpec instance given a python function and signature.

    Args:
      python_function: a function to inspect
      input_signature: a signature of the function (None, if variable)
      is_pure: if True all input arguments (including variables and constants)
        will be converted to tensors and no variable changes allowed.
      jit_compile: see `tf.function`

    Returns:
      instance of FunctionSpec
    """
    ...
  
  @classmethod
  def from_fullargspec_and_signature(cls, fullargspec, input_signature, is_pure=..., name=..., jit_compile=...): # -> FunctionSpec:
    """Construct FunctionSpec from legacy FullArgSpec format."""
    ...
  
  def __init__(self, function_type, default_values, is_pure=..., name=..., jit_compile=...) -> None:
    """Constructs a FunctionSpec describing a python function.

    Args:
      function_type: A FunctionType describing the python function signature.
      default_values: Dictionary mapping parameter names to default values.
      is_pure: if True all input arguments (including variables and constants)
        will be converted to tensors and no variable changes allowed.
      name: Name of the function
      jit_compile: see `tf.function`.
    """
    ...
  
  @property
  def default_values(self): # -> Any:
    """Returns dict mapping parameter names to default values."""
    ...
  
  @property
  def function_type(self): # -> Any:
    """Returns a FunctionType representing the Python function signature."""
    ...
  
  @property
  def fullargspec(self): # -> FullArgSpec:
    ...
  
  @property
  def input_signature(self): # -> tuple[Any, ...] | None:
    ...
  
  @property
  def flat_input_signature(self): # -> tuple[Any, ...]:
    ...
  
  @property
  def is_pure(self): # -> bool:
    ...
  
  @property
  def jit_compile(self): # -> None:
    ...
  
  @property
  def arg_names(self): # -> list[Any]:
    ...
  
  def signature_summary(self, default_values=...): # -> str:
    """Returns a string summarizing this function's signature.

    Args:
      default_values: If true, then include default values in the signature.

    Returns:
      A `string`.
    """
    ...
  


def make_function_type(python_function, input_signature): # -> tuple[FunctionType, Dict[str, Any]]:
  """Generates a FunctionType for python_function."""
  ...

def make_canonicalized_monomorphic_type(args: Any, kwargs: Any, capture_types: Any, polymorphic_type) -> Tuple[function_type_lib.FunctionType, trace_type.InternalTracingContext]:
  """Generates function type given the function arguments."""
  ...

def canonicalize_function_inputs(args, kwargs, function_type, default_values=..., is_pure=...):
  """Canonicalizes `args` and `kwargs`.

  Canonicalize the inputs to the Python function using FunctionType.
  In particular, we parse the varargs and kwargs that the
  original function was called with into a tuple corresponding to the
  Python function's positional (named) arguments and a dictionary
  corresponding to its kwargs.  Missing default arguments are added.

  If the FunctionType has an type constraints, then they are used to convert
  arguments to tensors; otherwise, any inputs containing numpy arrays are
  converted to tensors.


  Args:
    args: The varargs this object was called with.
    kwargs: The keyword args this function was called with.
    function_type: FunctionType to canonicalize against.
    default_values: Default values to use.
    is_pure: Force variable inputs to Tensors.

  Returns:
    A canonicalized ordering of the inputs, as well as full and filtered
    (Tensors and Variables only) versions of their concatenated flattened
    representations, represented by a tuple in the form (args, kwargs,
    flat_args, filtered_flat_args). Here: `args` is a full list of bound
    arguments, and `kwargs` contains only true keyword arguments, as opposed
    to named arguments called in a keyword-like fashion.

  Raises:
    ValueError: If a keyword in `kwargs` cannot be matched with a positional
      argument when an input signature is specified, or when the inputs
      do not conform to the input signature.
  """
  ...

def bind_function_inputs(args, kwargs, function_type, default_values):
  """Bind `args` and `kwargs` into a canonicalized signature args, kwargs."""
  ...

def derive_from_graph(func_graph): # -> FunctionType:
  """Derives a FunctionType from FuncGraph."""
  ...

def is_same_structure(structure1, structure2, check_values=...): # -> bool:
  """Check two structures for equality, optionally of types and of values."""
  ...

