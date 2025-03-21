"""
This type stub file was generated by pyright.
"""

from tensorflow.python.data.ops import dataset_ops

"""The implementation of `tf.data.Dataset.flat_map`."""
class _FlatMapDataset(dataset_ops.UnaryDataset):
  """A `Dataset` that maps a function over its input and flattens the result."""
  def __init__(self, input_dataset, map_func, name=...) -> None:
    ...
  
  @property
  def element_spec(self):
    ...
  


