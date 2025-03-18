"""
This type stub file was generated by pyright.
"""

import collections
import inspect
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple
from tensorflow.core.function import trace_type
from tensorflow.core.function.polymorphism import function_type_pb2
from tensorflow.python.types import core, trace

"""Represents the types of TF functions."""
class CapturedDefaultValue:
  def __repr__(self): # -> Literal['<captured_default_value>']:
    ...
  
  def __str__(self) -> str:
    ...
  


CAPTURED_DEFAULT_VALUE = ...
PROTO_TO_PY_ENUM = ...
PY_TO_PROTO_ENUM = ...
class Parameter(inspect.Parameter):
  """Represents a parameter to a function."""
  def __init__(self, name: str, kind: Any, optional: bool, type_constraint: Optional[trace.TraceType]) -> None:
    ...
  
  @classmethod
  def from_proto(cls, proto: Any) -> Parameter:
    """Generate a Parameter from the proto representation."""
    ...
  
  def to_proto(self) -> function_type_pb2.Parameter:
    """Generate a proto representation of the Parameter."""
    ...
  
  @property
  def optional(self) -> bool:
    """If this parameter might not be supplied for a call."""
    ...
  
  @property
  def type_constraint(self) -> Optional[trace.TraceType]:
    """A supertype that the parameter's type must subtype for validity."""
    ...
  
  def is_subtype_of(self, other: Parameter) -> bool:
    """Returns True if self is a supertype of other Parameter."""
    ...
  
  def most_specific_common_supertype(self, others: Sequence[Parameter]) -> Optional[Parameter]:
    """Returns a common supertype (if exists)."""
    ...
  
  def __eq__(self, other: Any) -> bool:
    ...
  
  def __hash__(self) -> int:
    ...
  
  def __repr__(self): # -> str:
    ...
  
  def __reduce__(self): # -> tuple[type[Self], tuple[str, _ParameterKind, bool, TraceType | None]]:
    ...
  


class FunctionType(core.FunctionType):
  """Represents the type of a TensorFlow function.

  FunctionType is the canonical way to represent the input/output contract of
  all kinds of functions within the tf.function domain, including:
    - Polymorphic Function
    - Concrete Function
    - Atomic Function

  It provides consistent, centralized and layered logic for:
    - Canonicalization of Python input arguments
    - Type-based dispatch to monomorphic functions
    - Packing/unpacking structured python values to Tensors
    - Generation of structured placeholder values for tracing

  Additionaly, it also provides:
    - Lossless serialization
    - Native integration with Python function signature representation
    - Seamless migration from older representation formats
  """
  def __init__(self, parameters: Sequence[inspect.Parameter], captures: Optional[collections.OrderedDict] = ..., **kwargs) -> None:
    ...
  
  @property
  def parameters(self) -> Mapping[str, Any]:
    """Returns an ordered mapping of parameter name to specification."""
    ...
  
  @property
  def captures(self) -> collections.OrderedDict:
    """Returns an ordered mapping of capture id to type."""
    ...
  
  @property
  def output(self) -> Optional[trace.TraceType]:
    """Return the output TraceType if specified."""
    ...
  
  @classmethod
  def from_callable(cls, obj: Callable[..., Any], *, follow_wrapped: bool = ...) -> FunctionType:
    """Generate FunctionType from a python Callable."""
    ...
  
  @classmethod
  def get_default_values(cls, obj: Callable[..., Any], *, follow_wrapped: bool = ...) -> Dict[str, Any]:
    """Inspects and returns a dictionary of default values."""
    ...
  
  @classmethod
  def from_proto(cls, proto: Any) -> FunctionType:
    """Generate a FunctionType from the proto representation."""
    ...
  
  def to_proto(self) -> Any:
    """Generate a proto representation from the FunctionType."""
    ...
  
  def bind_with_defaults(self, args, kwargs, default_values): # -> BoundArguments:
    """Returns BoundArguments with default values filled in."""
    ...
  
  def is_supertype_of(self, other: FunctionType) -> bool:
    """Returns True if self is a supertype of other FunctionType."""
    ...
  
  def most_specific_common_subtype(self, others: Sequence[FunctionType]) -> Optional[FunctionType]:
    """Returns a common subtype (if exists)."""
    ...
  
  def placeholder_arguments(self, placeholder_context: trace.PlaceholderContext) -> inspect.BoundArguments:
    """Returns BoundArguments of values that can be used for tracing."""
    ...
  
  @property
  def flat_inputs(self) -> List[trace.TraceType]:
    """Flat tensor inputs accepted by this FunctionType."""
    ...
  
  def unpack_inputs(self, bound_parameters: inspect.BoundArguments) -> List[core.Tensor]:
    """Unpacks python arguments to flat tensor inputs accepted by this type."""
    ...
  
  @property
  def flat_captures(self) -> List[trace.TraceType]:
    """Flat tensor captures needed by this FunctionType."""
    ...
  
  def unpack_captures(self, captures) -> List[core.Tensor]:
    """Unpacks captures to flat tensors."""
    ...
  
  @property
  def flat_outputs(self) -> List[trace.TraceType]:
    """Flat tensor outputs returned by this FunctionType."""
    ...
  
  def pack_output(self, flat_values: Sequence[core.Tensor]) -> Any:
    """Packs flat tensors to generate a value of the output type."""
    ...
  
  def __eq__(self, other: Any) -> bool:
    ...
  
  def __hash__(self) -> int:
    ...
  
  def __repr__(self): # -> str:
    ...
  


MAX_SANITIZATION_WARNINGS = ...
sanitization_warnings_given = ...
def sanitize_arg_name(name: str) -> str:
  """Sanitizes function argument names.

  Matches Python symbol naming rules.

  Without sanitization, names that are not legal Python parameter names can be
  set which makes it challenging to represent callables supporting the named
  calling capability.

  Args:
    name: The name to sanitize.

  Returns:
    A string that meets Python parameter conventions.
  """
  ...

def canonicalize_to_monomorphic(args: Tuple[Any, ...], kwargs: Dict[Any, Any], default_values: Dict[Any, Any], capture_types: collections.OrderedDict, polymorphic_type: FunctionType) -> Tuple[FunctionType, trace_type.InternalTracingContext]:
  """Generates a monomorphic type out of polymorphic type for given args."""
  ...

def add_type_constraints(function_type: FunctionType, input_signature: Any, default_values: Dict[str, Any]) -> FunctionType:
  """Adds type constraints to a FunctionType based on the input_signature."""
  ...

def from_structured_signature(input_signature=..., output_signature=..., capture_types=...) -> FunctionType:
  """Generates a FunctionType from legacy signature representation."""
  ...

def to_structured_signature(function_type: FunctionType) -> Tuple[Any, Any]:
  """Returns structured input and output signatures from a FunctionType."""
  ...

