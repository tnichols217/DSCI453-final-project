"""
This type stub file was generated by pyright.
"""

import sys
import astunparse

"""Converting code to AST.

Adapted from Tangent.
"""
PY2_PREAMBLE = ...
PY3_PREAMBLE = ...
MAX_SIZE = ...
if sys.version_info >= (3, 9):
  astunparse = ...
if sys.version_info >= (3, ):
  STANDARD_PREAMBLE = ...
  MAX_SIZE = ...
else:
  ...
STANDARD_PREAMBLE_LEN = ...
_LEADING_WHITESPACE = ...
def dedent_block(code_string): # -> LiteralString:
  """Dedents a code so that its first line starts at row zero."""
  ...

def parse_entity(entity, future_features): # -> tuple[Any, Any] | tuple[Any, str]:
  """Returns the AST and source code of given entity.

  Args:
    entity: Any, Python function/method/class
    future_features: Iterable[Text], future features to use (e.g.
      'print_statement'). See
      https://docs.python.org/2/reference/simple_stmts.html#future

  Returns:
    gast.AST, Text: the parsed AST node; the source code that was parsed to
    generate the AST (including any prefixes that this function may have added).
  """
  ...

def parse(src, preamble_len=..., single_node=...): # -> Any:
  """Returns the AST of given piece of code.

  Args:
    src: Text
    preamble_len: Int, indicates leading nodes in the parsed AST which should be
      dropped.
    single_node: Bool, whether `src` is assumed to be represented by exactly one
      AST node.

  Returns:
    ast.AST
  """
  ...

def parse_expression(src): # -> Any:
  """Returns the AST of given identifier.

  Args:
    src: A piece of code that represents a single Python expression
  Returns:
    A gast.AST object.
  Raises:
    ValueError: if src does not consist of a single Expression.
  """
  ...

def unparse(node, indentation=..., include_encoding_marker=...): # -> LiteralString:
  """Returns the source code of given AST.

  Args:
    node: The code to compile, as an AST object.
    indentation: Unused, deprecated. The returning code will always be indented
      at 4 spaces.
    include_encoding_marker: Bool, whether to include a comment on the first
      line to explicitly specify UTF-8 encoding.

  Returns:
    code: The source code generated from the AST object
    source_mapping: A mapping between the user and AutoGraph generated code.
  """
  ...

