"""
This type stub file was generated by pyright.
"""

import collections

"""Operators specific to slicing operations."""
class GetItemOpts(collections.namedtuple('GetItemOpts', ('element_dtype', ))):
  ...


def get_item(target, i, opts): # -> Any:
  """The slice read operator (i.e. __getitem__).

  Note: it is unspecified whether target will be mutated or not. In general,
  if target is mutable (like Python lists), it will be mutated.

  Args:
    target: An entity that supports getitem semantics.
    i: Index to read from.
    opts: A GetItemOpts object.

  Returns:
    The read element.

  Raises:
    ValueError: if target is not of a supported type.
  """
  ...

def set_item(target, i, x): # -> TensorArray | Any:
  """The slice write operator (i.e. __setitem__).

  Note: it is unspecified whether target will be mutated or not. In general,
  if target is mutable (like Python lists), it will be mutated.

  Args:
    target: An entity that supports setitem semantics.
    i: Index to modify.
    x: The new element value.

  Returns:
    Same as target, after the update was performed.

  Raises:
    ValueError: if target is not of a supported type.
  """
  ...

