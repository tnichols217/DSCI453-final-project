"""
This type stub file was generated by pyright.
"""

"""Ops for compressing and uncompressing dataset elements."""
def compress(element): # -> Any:
  """Compress a dataset element.

  Args:
    element: A nested structure of types supported by Tensorflow.

  Returns:
    A variant tensor representing the compressed element. This variant can be
    passed to `uncompress` to get back the original element.
  """
  ...

def uncompress(element, output_spec): # -> defaultdict[Any, Any] | Any | list[Any] | None:
  """Uncompress a compressed dataset element.

  Args:
    element: A scalar variant tensor to uncompress. The element should have been
      created by calling `compress`.
    output_spec: A nested structure of `tf.TypeSpec` representing the type(s) of
      the uncompressed element.

  Returns:
    The uncompressed element.
  """
  ...

