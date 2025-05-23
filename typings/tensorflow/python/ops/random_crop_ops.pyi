"""
This type stub file was generated by pyright.
"""

from tensorflow.python.util import deprecation, dispatch
from tensorflow.python.util.tf_export import tf_export

"""Operations for random tensor cropping."""
@tf_export("image.random_crop", v1=["image.random_crop", "random_crop"])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints("random_crop")
def random_crop(value, size, seed=..., name=...): # -> Any:
  """Randomly crops a tensor to a given size.

  Slices a shape `size` portion out of `value` at a uniformly chosen offset.
  Requires `value.shape >= size`.

  If a dimension should not be cropped, pass the full size of that dimension.
  For example, RGB images can be cropped with
  `size = [crop_height, crop_width, 3]`.

  Example usage:

  >>> image = [[1, 2, 3], [4, 5, 6]]
  >>> result = tf.image.random_crop(value=image, size=(1, 3))
  >>> result.shape.as_list()
  [1, 3]

  For producing deterministic results given a `seed` value, use
  `tf.image.stateless_random_crop`. Unlike using the `seed` param with
  `tf.image.random_*` ops, `tf.image.stateless_random_*` ops guarantee the same
  results given the same seed independent of how many times the function is
  called, and independent of global seed settings (e.g. tf.random.set_seed).

  Args:
    value: Input tensor to crop.
    size: 1-D tensor with size the rank of `value`.
    seed: Python integer. Used to create a random seed. See
      `tf.random.set_seed`
      for behavior.
    name: A name for this operation (optional).

  Returns:
    A cropped tensor of the same rank as `value` and shape `size`.
  """
  ...

@tf_export("image.stateless_random_crop", v1=[])
@dispatch.add_dispatch_support
def stateless_random_crop(value, size, seed, name=...): # -> Any:
  """Randomly crops a tensor to a given size in a deterministic manner.

  Slices a shape `size` portion out of `value` at a uniformly chosen offset.
  Requires `value.shape >= size`.

  If a dimension should not be cropped, pass the full size of that dimension.
  For example, RGB images can be cropped with
  `size = [crop_height, crop_width, 3]`.

  Guarantees the same results given the same `seed` independent of how many
  times the function is called, and independent of global seed settings (e.g.
  `tf.random.set_seed`).

  Usage Example:

  >>> image = [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]
  >>> seed = (1, 2)
  >>> tf.image.stateless_random_crop(value=image, size=(1, 2, 3), seed=seed)
  <tf.Tensor: shape=(1, 2, 3), dtype=int32, numpy=
  array([[[1, 2, 3],
          [4, 5, 6]]], dtype=int32)>

  Args:
    value: Input tensor to crop.
    size: 1-D tensor with size the rank of `value`.
    seed: A shape [2] Tensor, the seed to the random number generator. Must have
      dtype `int32` or `int64`. (When using XLA, only `int32` is allowed.)
    name: A name for this operation (optional).

  Returns:
    A cropped tensor of the same rank as `value` and shape `size`.
  """
  ...

