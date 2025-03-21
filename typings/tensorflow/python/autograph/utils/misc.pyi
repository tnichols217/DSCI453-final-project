"""
This type stub file was generated by pyright.
"""

"""Miscellaneous utilities that don't fit anywhere else."""
def alias_tensors(*args): # -> Generator[Any | defaultdict[Any, Any] | list[Any] | object | None, None, None] | defaultdict[Any, Any] | Any | list[Any] | object | None:
  """Wraps any Tensor arguments with an identity op.

  Any other argument, including Variables, is returned unchanged.

  Args:
    *args: Any arguments. Must contain at least one element.

  Returns:
    Same as *args, with Tensor instances replaced as described.

  Raises:
    ValueError: If args doesn't meet the requirements.
  """
  ...

def get_range_len(start, limit, delta):
  ...

