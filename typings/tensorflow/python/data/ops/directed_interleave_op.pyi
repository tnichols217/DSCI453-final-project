"""
This type stub file was generated by pyright.
"""

from tensorflow.python.data.ops import dataset_ops

"""The implementation of `tf.data.Dataset.shuffle`."""
class _DirectedInterleaveDataset(dataset_ops.DatasetV2):
  """A substitute for `Dataset.interleave()` on a fixed list of datasets."""
  def __init__(self, selector_input, data_inputs, stop_on_empty_dataset=...) -> None:
    ...
  
  @property
  def element_spec(self): # -> defaultdict[Any, Any] | Any | list[Any] | object | None:
    ...
  


