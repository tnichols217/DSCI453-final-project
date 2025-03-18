"""
This type stub file was generated by pyright.
"""

import contextlib
from tensorflow.python.ops import resource_variable_ops

"""Utility functions for TPU."""
def enclosing_tpu_context(): # -> TPUReplicateContext | None:
  """Returns the TPUReplicateContext, which exists inside a tpu.rewrite()."""
  ...

def enclosing_tpu_context_and_graph(): # -> tuple[TPUReplicateContext, Graph | Any] | tuple[None, None]:
  """Returns the TPUReplicateContext which exists inside a tpu.rewrite(), and its associated graph."""
  ...

@contextlib.contextmanager
def outside_or_skip_tpu_context(): # -> Generator[None, Any, None]:
  """Returns a context manager that skips current enclosing context if there is any."""
  ...

def make_raw_assign_fn(raw_assign_fn, use_handle=...): # -> Callable[..., Any]:
  """Wrap `raw_assign_fn` with the proper graph context and device scope.

  Args:
    raw_assign_fn: the function to be wrapped.
    use_handle: if True, the `raw_assign_fn` will be applied to the handle of a
      variable; otherwise it will be applied to the variable itself.

  Returns:
    The wrapped function.
  """
  ...

def make_raw_scatter_xxx_fn(raw_scatter_xxx_fn): # -> Callable[..., Any]:
  """Wrap `raw_scatter_xxx_fn` so that it can be called w/ and w/o packed handle."""
  ...

class LazyVariableTracker:
  """Class to track uninitialized lazy variables."""
  def __init__(self) -> None:
    ...
  
  def initialize_all(self): # -> None:
    """Initialize all uninitialized lazy variables stored in scope."""
    ...
  
  def add_uninitialized_var(self, var): # -> None:
    ...
  


class TPUUninitializedVariable(resource_variable_ops.UninitializedVariable):
  """UninitializedVariable component for TPU.

  Sometimes user might assign (different values) to a single component of a
  mirrored TPU variable. Thus we need to initialize_all when the assign* or read
  is invoked on a single component.
  """
  def read_value(self): # -> defaultdict[Any, Any] | Any | list[Any] | object | None:
    ...
  
  def assign_sub(self, delta, use_locking=..., name=..., read_value=...): # -> _UnreadVariable | object | Operation | None:
    ...
  
  def assign(self, value, use_locking=..., name=..., read_value=...): # -> _UnreadVariable | object | Operation | None:
    ...
  
  def assign_add(self, delta, use_locking=..., name=..., read_value=...): # -> _UnreadVariable | object | Operation | None:
    ...
  


