"""
This type stub file was generated by pyright.
"""

from tensorflow.python.data.ops import dataset_ops

"""The implementation of `tf.data.Dataset.padded_batch`."""
class _PaddedBatchDataset(dataset_ops.UnaryDataset):
  """A `Dataset` that batches and pads contiguous elements from its input."""
  def __init__(self, input_dataset, batch_size, padded_shapes, padding_values, drop_remainder, name=...) -> None:
    """See `Dataset.batch()` for details."""
    ...
  
  @property
  def element_spec(self): # -> defaultdict[Any, Any] | Any | list[Any] | None:
    ...
  


