"""
This type stub file was generated by pyright.
"""

"""Exception handling statements: assert, etc."""
def assert_stmt(expression1, expression2): # -> object | Operation | Any | None:
  """Functional form of an assert statement.

  This follows the semantics of the Python assert statement, however the
  concrete implementations may deviate from it. See the respective
  implementation for details.

  In general, the assert statement should not be used for control flow.
  Furthermore, it is encouraged that the assertion expressions should not have
  side effects.

  Args:
    expression1: Any
    expression2: Callable[[], Any], returns the expression to include in the
        error message when expression1 evaluates to False. When expression1 is
        True, the result of expression2 will not be evaluated, however,
        expression2 itself may be evaluated in some implementations.

  Returns:
    Any, implementation-dependent.

  Raises:
    ValueError: if any arguments are illegal.
  """
  ...

