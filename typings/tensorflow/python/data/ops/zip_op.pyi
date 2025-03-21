"""
This type stub file was generated by pyright.
"""

from tensorflow.python.data.ops import dataset_ops

"""The implementation of `tf.data.Dataset.zip`."""
class _ZipDataset(dataset_ops.DatasetV2):
  """A `Dataset` that zips its inputs together."""
  def __init__(self, datasets, name=...) -> None:
    """See `Dataset.zip()` for details."""
    ...
  
  @property
  def element_spec(self): # -> defaultdict[Any, Any] | Any | list[Any] | None:
    ...
  


