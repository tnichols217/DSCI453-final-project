"""
This type stub file was generated by pyright.
"""

from tensorflow.python.data.ops import dataset_ops

"""The implementation of `tf.data.Dataset.shuffle`."""
class _ScanDataset(dataset_ops.UnaryDataset):
  """A dataset that scans a function across its input."""
  def __init__(self, input_dataset, initial_state, scan_func, use_default_device=..., name=...) -> None:
    """See `scan()` for details."""
    ...
  
  @property
  def element_spec(self): # -> defaultdict[Any, Any] | Any | list[Any] | None:
    ...
  


