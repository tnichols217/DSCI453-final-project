"""
This type stub file was generated by pyright.
"""

import abc
from tensorflow.python.util.tf_export import tf_export

"""Dataset types."""
@tf_export("__internal__.types.data.Dataset", v1=[])
class DatasetV2(abc.ABC):
  """Represents the TensorFlow 2 type `tf.data.Dataset`."""
  ...


@tf_export(v1=["__internal__.types.data.Dataset"])
class DatasetV1(DatasetV2, abc.ABC):
  """Represents the TensorFlow 1 type `tf.data.Dataset`."""
  ...


