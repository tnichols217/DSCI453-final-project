"""
This type stub file was generated by pyright.
"""

import types

"""A LazyLoader class."""
_TENSORFLOW_LAZY_LOADER_PREFIX = ...
class LazyLoader(types.ModuleType):
  """Lazily import a module, mainly to avoid pulling in large dependencies.

  `contrib`, and `ffmpeg` are examples of modules that are large and not always
  needed, and this allows them to only be loaded when they are used.
  """
  def __init__(self, local_name, parent_module_globals, name, warning=...) -> None:
    ...
  
  def __getattr__(self, name): # -> Any:
    ...
  
  def __setattr__(self, name, value): # -> None:
    ...
  
  def __delattr__(self, name): # -> None:
    ...
  
  def __repr__(self): # -> str:
    ...
  
  def __dir__(self): # -> list[str]:
    ...
  
  def __reduce__(self): # -> tuple[Callable[..., ModuleType], tuple[str]]:
    ...
  


class KerasLazyLoader(LazyLoader):
  """LazyLoader that handles routing to different Keras version."""
  def __init__(self, parent_module_globals, mode=..., submodule=..., name=...) -> None:
    ...
  
  def __getattr__(self, item): # -> Any:
    ...
  
  def __repr__(self): # -> str:
    ...
  
  def __dir__(self): # -> list[str]:
    ...
  


