"""
This type stub file was generated by pyright.
"""

from tensorflow.python.autograph.core import converter

"""Canonicalizes continue statements by de-sugaring into a control boolean."""
class _Continue:
  def __init__(self) -> None:
    ...
  
  def __repr__(self): # -> str:
    ...
  


class _Block:
  """Tracks information about lexical blocks as they are visited in the AST.

  Mainly, this object tracks the creation of block guards that replace
  `continue` statements (e.g. `if not continue_:`).

  Attributes:
    create_guard_current: bool, whether to create a guard for the current
      statement.
    create_guard_next: bool, whether to create a guard for the next
      statement.
    is_loop_type: bool, whether this block is the body of a loop.
  """
  def __init__(self) -> None:
    ...
  


class ContinueCanonicalizationTransformer(converter.Base):
  """Canonicalizes continue statements into additional conditionals."""
  def visit_Continue(self, node): # -> list[Any]:
    ...
  
  def visit_While(self, node): # -> While:
    ...
  
  def visit_For(self, node): # -> For:
    ...
  
  def visit_If(self, node): # -> If:
    ...
  
  def visit_With(self, node): # -> With:
    ...
  
  def visit_Try(self, node): # -> Try:
    ...
  
  def visit_ExceptHandler(self, node): # -> ExceptHandler:
    ...
  


def transform(node, ctx): # -> AST | list[Any] | tuple[Any, ...] | Any:
  ...

