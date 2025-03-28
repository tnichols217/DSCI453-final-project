"""
This type stub file was generated by pyright.
"""

from typing import Any, Callable
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import tensor_spec

"""Implementation of LoadDataset in Python."""
_RETRY_INTERVAL_SEC = ...
class _LoadDataset(dataset_ops.DatasetSource):
  """A dataset that loads previously saved dataset."""
  def __init__(self, path: str, element_spec: Any, compression: str, reader_func: Callable[[dataset_ops.Dataset], dataset_ops.Dataset]) -> None:
    ...
  
  @property
  def element_spec(self) -> Any:
    ...
  


class _SnapshotChunkDataset(dataset_ops.DatasetSource):
  """A dataset for one chunk file from a tf.data distributed snapshot."""
  def __init__(self, chunk_file: str, element_spec: Any, compression: str) -> None:
    ...
  
  @property
  def element_spec(self) -> Any:
    ...
  


class _ListSnapshotChunksDataset(dataset_ops.DatasetSource):
  """A dataset for listing snapshot chunk files.

  It supports listing partially written snapshots. When a snapshot is being
  written, it returns the currently available chunk files.
  """
  def __init__(self, snapshot_path: str) -> None:
    ...
  
  @property
  def element_spec(self) -> tensor_spec.TensorSpec:
    ...
  


