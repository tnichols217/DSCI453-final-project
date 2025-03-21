"""
This type stub file was generated by pyright.
"""

from tensorflow.python.data.ops import dataset_ops

"""The implementation of `tf.data.Dataset.rebatch`."""
class _RebatchDataset(dataset_ops.UnaryDataset):
  """A `Dataset` that rebatches elements from its input into new batch sizes.

  `_RebatchDataset(input_dataset, batch_sizes)` is functionally equivalent to
  `input_dataset.unbatch().batch(N)`, where the value of N cycles through the
  `batch_sizes` input list. The elements produced by this dataset have the same
  rank as the elements of the input dataset.
  """
  def __init__(self, input_dataset, batch_sizes, drop_remainder=..., name=...) -> None:
    """See `Dataset.rebatch` for details."""
    ...
  
  @property
  def element_spec(self): # -> defaultdict[Any, Any] | Any | list[Any] | object | None:
    ...
  


