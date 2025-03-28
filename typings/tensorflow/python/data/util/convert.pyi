"""
This type stub file was generated by pyright.
"""

"""Helpers constructing Datasets."""
def optional_param_to_tensor(argument_name, argument_value, argument_default=..., argument_dtype=...): # -> SymbolicTensor | Operation | _EagerTensorBase:
  ...

def partial_shape_to_tensor(shape_like): # -> SymbolicTensor:
  """Returns a `tf.Tensor` that represents the given shape.

  Args:
    shape_like: A value that can be converted to a `tf.TensorShape` or a
      `tf.Tensor`.

  Returns:
    A 1-D `tf.Tensor` of `tf.int64` elements representing the given shape, where
    `-1` is substituted for any unknown dimensions.
  """
  ...

