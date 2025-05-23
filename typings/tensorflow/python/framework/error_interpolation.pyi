"""
This type stub file was generated by pyright.
"""

"""Function for interpolating formatted errors from the TensorFlow runtime.

Exposes the function `interpolate` to interpolate messages with tags of the form
{{type name}}.
"""
_NAME_REGEX = ...
_TAG_REGEX = ...
_INTERPOLATION_REGEX = ...
_INTERPOLATION_PATTERN = ...
_ParseTag = ...
_FRAMEWORK_COMMON_PREFIX = ...
_FRAMEWORK_PATH_PREFIXES = ...
_FRAMEWORK_FILENAME_PATTERNS = ...
_EXTERNAL_FILENAME_PATTERNS = ...
def parse_message(message): # -> tuple[LiteralString, list[Any], list[Any]]:
  """Extract function tags and node tags from a message.

  Tags are named tuples representing the string {{type name}}. For example,
  in "123{{node Foo}}456{{function_node Bar}}789", there are two tags: a node
  tag and a function tag.

  Args:
    message: An error message, possibly from an OpError.

  Returns:
    A tuple containing the original message with function nodes stripped,
    function tags, and node tags.

    For example, if message is "123{{node Foo}}456{{function_node Bar}}789"
    then this function returns ("123{{node Foo}}456789",
    [_ParseTag("function_node", "Bar")], [_ParseTag("node", "Foo")]).
  """
  ...

def create_graph_debug_info_def(func_named_operations):
  """Construct and returns a `GraphDebugInfo` protocol buffer.

  Args:
    func_named_operations: An iterable of (func_name, op.Operation) tuples
      where the Operation instances have a _traceback members. The func_name
      should be the empty string for operations in the top-level Graph.

  Returns:
    GraphDebugInfo protocol buffer.

  Raises:
    TypeError: If the arguments are not of the correct proto buffer type.
  """
  ...

def interpolate_graph(message, graph): # -> str:
  """Interpolates an error message.

  The error message can contain tags of form `{{node_type node_name}}`
  which will be parsed to identify the tf.Graph and op. If the op contains
  traceback, the traceback will be attached to the error message.

  Args:
    message: A string to interpolate.
    graph: ops.Graph object containing all nodes referenced in the error
        message.

  Returns:
    The error message string with node definition traceback.
  """
  ...

