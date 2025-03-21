"""
This type stub file was generated by pyright.
"""

"""Schema utilities to get builtin code from operator code."""
def get_builtin_code_from_operator_code(opcode):
  """Return the builtin code of the given operator code.

  The following method is introduced to resolve op builtin code shortage
  problem. The new builtin operator will be assigned to the extended builtin
  code field in the flatbuffer schema. Those methods helps to hide builtin code
  details.

  Args:
    opcode: Operator code.

  Returns:
    The builtin code of the given operator code.
  """
  ...

_allowed_symbols = ...
