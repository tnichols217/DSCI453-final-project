"""
This type stub file was generated by pyright.
"""

import collections as py_collections
from typing import Any, Hashable, Mapping, Optional
from tensorflow.python.types import core

"""FuncGraph and related functionality."""
_EAGER_CONST_THRESHOLD = ...
class MutationAwareDict(py_collections.OrderedDict):
  """A dict with a mutation flag."""
  def __init__(self, *args, **kwargs) -> None:
    ...
  
  def pop(self, key, default=...):
    ...
  
  def __setitem__(self, key, value): # -> None:
    ...
  
  def __delitem__(self, key): # -> None:
    ...
  
  def clear(self): # -> None:
    ...
  
  @property
  def mutated(self): # -> bool:
    ...
  
  @mutated.setter
  def mutated(self, value): # -> None:
    ...
  


class FunctionCaptures:
  """A container for all capture usages within FuncGraph."""
  def __init__(self) -> None:
    ...
  
  def clear(self): # -> None:
    ...
  
  def capture_by_value(self, graph: Any, tensor: core.Tensor, name: Optional[str] = ...) -> core.Tensor:
    """Captures `tensor` if it's external to this graph.

    If `tensor` is from a different graph, returns a placeholder for it.
    `tensor` and the placeholder will appear in self.captures, and the
    placeholder will appear in self.inputs.  Multiple calls to this method with
    the same `tensor` argument will return the same placeholder. If `tensor` is
    from this graph, returns `tensor`.

    Args:
      graph: The FuncGraph that captures this tensor.
      tensor: Tensor. May be from this FuncGraph or a different graph.
      name: Optional name if a placeholder is created.

    Returns:
      Tensor from this FuncGraph.

    Raises:
      InaccessibleTensorError: if any tensors are accessed in a manner that
      bypasses the mechanisms required for the data dependencies to be correctly
      wired.
    """
    ...
  
  def add_or_replace(self, key: Hashable, external: Any, internal: core.Tensor, tracetype: Any = ..., is_by_ref: bool = ...) -> None:
    """Replace a already exsiting capture, otherwise add it."""
    ...
  
  def pop(self, key: Hashable, is_by_ref: bool = ...) -> Any:
    ...
  
  def reset_captures(self, tensors, placeholders): # -> None:
    """Set the captures with the provided list of captures & placeholder."""
    ...
  
  def merge_by_ref_with(self, other: FunctionCaptures) -> None:
    """Add by-ref captures from `other` to `self` if not exist."""
    ...
  
  def get_by_ref_snapshot(self) -> Mapping[Hashable, Any]:
    """Get a snapshot of current values of by-ref captures."""
    ...
  
  @property
  def capture_types(self): # -> OrderedDict[Any, Any]:
    ...
  
  @property
  def by_val_capture_tuples(self): # -> list[Any]:
    ...
  
  @property
  def by_ref_internal(self): # -> OrderedDict[Any, Any]:
    ...
  
  @property
  def by_ref_external(self): # -> OrderedDict[Any, Any]:
    ...
  
  @property
  def by_ref_tracetype(self): # -> OrderedDict[Any, Any]:
    ...
  
  @property
  def by_val_internal(self): # -> MutationAwareDict:
    ...
  
  @property
  def by_val_external(self): # -> MutationAwareDict:
    ...
  
  @property
  def by_val_tracetype(self): # -> OrderedDict[Any, Any] | MutationAwareDict:
    ...
  


