"""
This type stub file was generated by pyright.
"""

"""Utility to retrieve function args."""
def fn_args(fn): # -> tuple[str, ...]:
  """Get argument names for function-like object.

  Args:
    fn: Function, or function-like object (e.g., result of `functools.partial`).

  Returns:
    `tuple` of string argument names.

  Raises:
    ValueError: if partial function has positionally bound arguments
  """
  ...

def has_kwargs(fn): # -> bool:
  """Returns whether the passed callable has **kwargs in its signature.

  Args:
    fn: Function, or function-like object (e.g., result of `functools.partial`).

  Returns:
    `bool`: if `fn` has **kwargs in its signature.

  Raises:
     `TypeError`: If fn is not a Function, or function-like object.
  """
  ...

def get_func_name(func): # -> str | LiteralString:
  """Returns name of passed callable."""
  ...

def get_func_code(func): # -> CodeType | Any | None:
  """Returns func_code of passed callable, or None if not available."""
  ...

_rewriter_config_optimizer_disabled = ...
def get_disabled_rewriter_config():
  ...

