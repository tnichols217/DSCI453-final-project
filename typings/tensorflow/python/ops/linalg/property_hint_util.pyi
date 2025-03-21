"""
This type stub file was generated by pyright.
"""

"""Common utilities for LinearOperator property hints."""
def combined_commuting_self_adjoint_hint(operator_a, operator_b): # -> bool | None:
  """Get combined hint for self-adjoint-ness."""
  ...

def is_square(operator_a, operator_b): # -> bool | None:
  """Return a hint to whether the composition is square."""
  ...

def combined_commuting_positive_definite_hint(operator_a, operator_b): # -> Literal[True] | None:
  """Get combined PD hint for compositions."""
  ...

def combined_non_singular_hint(operator_a, operator_b): # -> Literal[False]:
  """Get combined hint for when ."""
  ...

