"""
This type stub file was generated by pyright.
"""

from typing import Any, NamedTuple, Optional
from tensorflow.core.function.polymorphism import function_type as function_type_lib

"""Cache to manage functions based on their FunctionType."""
class FunctionContext(NamedTuple):
  """Contains information regarding tf.function execution context."""
  context: Any = ...
  scope_type: Any = ...


class FunctionCache:
  """A container for managing functions."""
  __slots__ = ...
  def __init__(self) -> None:
    ...
  
  def lookup(self, function_type: function_type_lib.FunctionType, context: Optional[FunctionContext] = ...) -> Optional[Any]:
    """Looks up a function based on the context and type."""
    ...
  
  def delete(self, function_type: function_type_lib.FunctionType, context: Optional[FunctionContext] = ...) -> bool:
    """Deletes a function given the context and type."""
    ...
  
  def add(self, fn: Any, context: Optional[FunctionContext] = ...) -> None:
    """Adds a new function using its function_type.

    Args:
      fn: The function to be added to the cache.
      context: A FunctionContext representing the current context.
    """
    ...
  
  def generalize(self, context: FunctionContext, function_type: function_type_lib.FunctionType) -> function_type_lib.FunctionType:
    """Try to generalize a FunctionType within a FunctionContext."""
    ...
  
  def clear(self): # -> None:
    """Removes all functions from the cache."""
    ...
  
  def values(self): # -> list[Any]:
    """Returns a list of all functions held by this cache."""
    ...
  
  def __len__(self): # -> int:
    ...
  


