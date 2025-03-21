"""
This type stub file was generated by pyright.
"""

from tensorflow.python.data.ops import dataset_ops

"""The implementation of `tf.data.Dataset.take_while`."""
class _TakeWhileDataset(dataset_ops.UnaryUnchangedStructureDataset):
  """A dataset that stops iteration when `predicate` returns false."""
  def __init__(self, input_dataset, predicate, name=...) -> None:
    """See `take_while()` for details."""
    ...
  


