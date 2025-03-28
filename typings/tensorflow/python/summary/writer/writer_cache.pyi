"""
This type stub file was generated by pyright.
"""

from tensorflow.python.util.tf_export import tf_export

"""A cache for FileWriters."""
@tf_export(v1=['summary.FileWriterCache'])
class FileWriterCache:
  """Cache for file writers.

  This class caches file writers, one per directory.
  """
  _cache = ...
  _lock = ...
  @staticmethod
  def clear(): # -> None:
    """Clear cached summary writers. Currently only used for unit tests."""
    ...
  
  @staticmethod
  def get(logdir):
    """Returns the FileWriter for the specified directory.

    Args:
      logdir: str, name of the directory.

    Returns:
      A `FileWriter`.
    """
    ...
  


