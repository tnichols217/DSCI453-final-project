"""
This type stub file was generated by pyright.
"""

from collections.abc import Iterator
from typing import Generic, Optional, TypeVar

"""A simple stack that associates filename and line numbers with each object."""
T = TypeVar("T")
class TraceableObject(Generic[T]):
  """Wrap an object together with its the code definition location."""
  def __init__(self, obj: T, filename: Optional[str] = ..., lineno: Optional[int] = ...) -> None:
    ...
  
  def set_filename_and_line_from_caller(self, offset: int = ...) -> int:
    """Set filename and line using the caller's stack frame.

    If the requested stack information is not available, a heuristic may
    be applied and self.HEURISTIC USED will be returned.  If the heuristic
    fails then no change will be made to the filename and lineno members
    (None by default) and self.FAILURE will be returned.

    Args:
      offset: Integer.  If 0, the caller's stack frame is used.  If 1,
          the caller's caller's stack frame is used.  Larger values are
          permissible but if out-of-range (larger than the number of stack
          frames available) the outermost stack frame will be used.

    Returns:
      TraceableObject.SUCCESS if appropriate stack information was found,
      TraceableObject.HEURISTIC_USED if the offset was larger than the stack,
      and TraceableObject.FAILURE if the stack was empty.
    """
    ...
  
  def copy_metadata(self): # -> Self:
    """Return a TraceableObject like this one, but without the object."""
    ...
  


class TraceableStack(Generic[T]):
  """A stack of TraceableObjects."""
  def __init__(self, existing_stack: Optional[list[TraceableObject[T]]] = ...) -> None:
    """Constructor.

    Args:
      existing_stack: [TraceableObject, ...] If provided, this object will
        set its new stack to a SHALLOW COPY of existing_stack.
    """
    ...
  
  def push_obj(self, obj: T, offset: int = ...): # -> int:
    """Add object to the stack and record its filename and line information.

    Args:
      obj: An object to store on the stack.
      offset: Integer.  If 0, the caller's stack frame is used.  If 1,
          the caller's caller's stack frame is used.

    Returns:
      TraceableObject.SUCCESS if appropriate stack information was found,
      TraceableObject.HEURISTIC_USED if the stack was smaller than expected,
      and TraceableObject.FAILURE if the stack was empty.
    """
    ...
  
  def pop_obj(self) -> T:
    """Remove last-inserted object and return it, without filename/line info."""
    ...
  
  def peek_top_obj(self) -> T:
    """Return the most recent stored object."""
    ...
  
  def peek_objs(self) -> Iterator[T]:
    """Return iterator over stored objects ordered newest to oldest."""
    ...
  
  def peek_traceable_objs(self) -> Iterator[TraceableObject[T]]:
    """Return iterator over stored TraceableObjects ordered newest to oldest."""
    ...
  
  def __len__(self) -> int:
    """Return number of items on the stack, and used for truth-value testing."""
    ...
  
  def copy(self) -> TraceableStack[T]:
    """Return a copy of self referencing the same objects but in a new list.

    This method is implemented to support thread-local stacks.

    Returns:
      TraceableStack with a new list that holds existing objects.
    """
    ...
  


