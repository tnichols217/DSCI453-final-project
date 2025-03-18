"""
This type stub file was generated by pyright.
"""

from tensorflow.python.framework import tensor

"""ndarray class."""
def convert_to_tensor(value, dtype=..., dtype_hint=...): # -> Tensor:
  """Wrapper over `tf.convert_to_tensor`.

  Args:
    value: value to convert
    dtype: (optional) the type we would like it to be converted to.
    dtype_hint: (optional) soft preference for the type we would like it to be
      converted to. `tf.convert_to_tensor` will attempt to convert value to this
      type first, but will not fail if conversion is not possible falling back
      to inferring the type instead.

  Returns:
    Value converted to tf.Tensor.
  """
  ...

ndarray = tensor.Tensor
