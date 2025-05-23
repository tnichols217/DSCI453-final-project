"""
This type stub file was generated by pyright.
"""

import collections
from typing import Any, Set

"""Utilities for collecting objects based on "is" comparison."""
class _ObjectIdentityWrapper:
  """Wraps an object, mapping __eq__ on wrapper to "is" on wrapped.

  Since __eq__ is based on object identity, it's safe to also define __hash__
  based on object ids. This lets us add unhashable types like trackable
  _ListWrapper objects to object-identity collections.
  """
  __slots__ = ...
  def __init__(self, wrapped) -> None:
    ...
  
  @property
  def unwrapped(self): # -> Any:
    ...
  
  def __lt__(self, other) -> bool:
    ...
  
  def __gt__(self, other) -> bool:
    ...
  
  def __eq__(self, other) -> bool:
    ...
  
  def __ne__(self, other) -> bool:
    ...
  
  def __hash__(self) -> int:
    ...
  
  def __repr__(self): # -> str:
    ...
  


class _WeakObjectIdentityWrapper(_ObjectIdentityWrapper):
  __slots__ = ...
  def __init__(self, wrapped) -> None:
    ...
  
  @property
  def unwrapped(self):
    ...
  


class Reference(_ObjectIdentityWrapper):
  """Reference that refers an object.

  ```python
  x = [1]
  y = [1]

  x_ref1 = Reference(x)
  x_ref2 = Reference(x)
  y_ref2 = Reference(y)

  print(x_ref1 == x_ref2)
  ==> True

  print(x_ref1 == y)
  ==> False
  ```
  """
  __slots__ = ...
  unwrapped = ...
  def deref(self): # -> Any:
    """Returns the referenced object.

    ```python
    x_ref = Reference(x)
    print(x is x_ref.deref())
    ==> True
    ```
    """
    ...
  


class ObjectIdentityDictionary(collections.abc.MutableMapping):
  """A mutable mapping data structure which compares using "is".

  This is necessary because we have trackable objects (_ListWrapper) which
  have behavior identical to built-in Python lists (including being unhashable
  and comparing based on the equality of their contents by default).
  """
  __slots__ = ...
  def __init__(self) -> None:
    ...
  
  def __getitem__(self, key):
    ...
  
  def __setitem__(self, key, value): # -> None:
    ...
  
  def __delitem__(self, key): # -> None:
    ...
  
  def __len__(self): # -> int:
    ...
  
  def __iter__(self): # -> Generator[Any, Any, None]:
    ...
  
  def __repr__(self): # -> str:
    ...
  


class ObjectIdentityWeakKeyDictionary(ObjectIdentityDictionary):
  """Like weakref.WeakKeyDictionary, but compares objects with "is"."""
  __slots__ = ...
  def __len__(self): # -> int:
    ...
  
  def __iter__(self): # -> Generator[Any, Any, None]:
    ...
  


class ObjectIdentitySet(collections.abc.MutableSet):
  """Like the built-in set, but compares objects with "is"."""
  __slots__ = ...
  def __init__(self, *args) -> None:
    ...
  
  def __le__(self, other: Set[Any]) -> bool:
    ...
  
  def __ge__(self, other: Set[Any]) -> bool:
    ...
  
  def __contains__(self, key): # -> bool:
    ...
  
  def discard(self, key): # -> None:
    ...
  
  def add(self, key): # -> None:
    ...
  
  def update(self, items): # -> None:
    ...
  
  def clear(self): # -> None:
    ...
  
  def intersection(self, items): # -> set[_ObjectIdentityWrapper]:
    ...
  
  def difference(self, items): # -> ObjectIdentitySet:
    ...
  
  def __len__(self): # -> int:
    ...
  
  def __iter__(self): # -> Generator[Any, Any, None]:
    ...
  


class ObjectIdentityWeakSet(ObjectIdentitySet):
  """Like weakref.WeakSet, but compares objects with "is"."""
  __slots__ = ...
  def __len__(self): # -> int:
    ...
  
  def __iter__(self): # -> Generator[Any, Any, None]:
    ...
  


