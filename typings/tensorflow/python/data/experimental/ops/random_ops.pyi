"""
This type stub file was generated by pyright.
"""

import functools
from tensorflow.python import tf2
from tensorflow.python.data.ops import dataset_ops, random_op
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export

"""Datasets for random number generators."""
@deprecation.deprecated(None, "Use `tf.data.Dataset.random(...)`.")
@tf_export("data.experimental.RandomDataset", v1=[])
class RandomDatasetV2(random_op._RandomDataset):
  """A `Dataset` of pseudorandom values."""
  ...


@deprecation.deprecated(None, "Use `tf.data.Dataset.random(...)`.")
@tf_export(v1=["data.experimental.RandomDataset"])
class RandomDatasetV1(dataset_ops.DatasetV1Adapter):
  """A `Dataset` of pseudorandom values."""
  @functools.wraps(RandomDatasetV2.__init__)
  def __init__(self, seed=...) -> None:
    ...
  


if tf2.enabled():
  RandomDataset = ...
else:
  RandomDataset = ...
