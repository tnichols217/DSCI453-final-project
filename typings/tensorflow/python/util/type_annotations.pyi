"""
This type stub file was generated by pyright.
"""

import typing

"""Utilities for accessing Python generic type annotations (typing.*)."""
def is_generic_union(tp): # -> bool:
  """Returns true if `tp` is a parameterized typing.Union value."""
  ...

def is_generic_tuple(tp): # -> bool:
  """Returns true if `tp` is a parameterized typing.Tuple value."""
  ...

def is_generic_list(tp): # -> bool:
  """Returns true if `tp` is a parameterized typing.List value."""
  ...

def is_generic_mapping(tp): # -> bool:
  """Returns true if `tp` is a parameterized typing.Mapping value."""
  ...

def is_forward_ref(tp): # -> bool:
  """Returns true if `tp` is a typing forward reference."""
  ...

if hasattr(typing, 'get_args'):
  get_generic_type_args = ...
else:
  get_generic_type_args = ...
