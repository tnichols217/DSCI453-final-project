"""
This type stub file was generated by pyright.
"""

from tensorflow.python.data.ops import dataset_ops

"""The implementation of `tf.data.Dataset.window`."""
class _WindowDataset(dataset_ops.UnaryDataset):
  """A dataset that creates window datasets from the input elements."""
  def __init__(self, input_dataset, size, shift, stride, drop_remainder, name=...) -> None:
    """See `window()` for more details."""
    ...
  
  @property
  def element_spec(self): # -> defaultdict[Any, Any] | Any | list[Any] | None:
    ...
  


