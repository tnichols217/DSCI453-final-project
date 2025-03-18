"""
This type stub file was generated by pyright.
"""

"""Provides wrapper for TensorFlow modules."""
FastModuleType = ...
_PER_MODULE_WARNING_LIMIT = ...
compat_v1_usage_gauge = ...
def get_rename_v2(name): # -> str | None:
  ...

def contains_deprecation_decorator(decorators): # -> bool:
  ...

def has_deprecation_decorator(symbol): # -> bool:
  """Checks if given object has a deprecation decorator.

  We check if deprecation decorator is in decorators as well as
  whether symbol is a class whose __init__ method has a deprecation
  decorator.
  Args:
    symbol: Python object.

  Returns:
    True if symbol has deprecation decorator.
  """
  ...

class TFModuleWrapper(FastModuleType):
  """Wrapper for TF modules to support deprecation messages and lazyloading."""
  compat_v1_usage_recorded = ...
  def __init__(self, wrapped, module_name, public_apis=..., deprecation=..., has_lite=...) -> None:
    ...
  
  def __setattr__(self, arg, val): # -> None:
    ...
  
  def __dir__(self): # -> list[Any | str] | list[str]:
    ...
  
  def __delattr__(self, name): # -> None:
    ...
  
  def __repr__(self):
    ...
  
  def __reduce__(self): # -> tuple[Callable[..., ModuleType], tuple[Any]]:
    ...
  


