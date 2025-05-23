"""
This type stub file was generated by pyright.
"""

import collections.abc

"""Immutable mapping."""
class ImmutableDict(collections.abc.Mapping):
  """Immutable `Mapping`."""
  def __init__(self, *args, **kwargs) -> None:
    ...
  
  def __getitem__(self, key):
    ...
  
  def __contains__(self, key): # -> bool:
    ...
  
  def __iter__(self): # -> Iterator[Any]:
    ...
  
  def __len__(self): # -> int:
    ...
  
  def __repr__(self): # -> str:
    ...
  
  __supported_by_tf_nest__ = ...


