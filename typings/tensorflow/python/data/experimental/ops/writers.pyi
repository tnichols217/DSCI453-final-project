"""
This type stub file was generated by pyright.
"""

from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export

"""Python wrappers for tf.data writers."""
@tf_export("data.experimental.TFRecordWriter")
@deprecation.deprecated(None, "To write TFRecords to disk, use `tf.io.TFRecordWriter`. To save " "and load the contents of a dataset, use `tf.data.experimental.save` " "and `tf.data.experimental.load`")
class TFRecordWriter:
  """Writes a dataset to a TFRecord file.

  The elements of the dataset must be scalar strings. To serialize dataset
  elements as strings, you can use the `tf.io.serialize_tensor` function.

  ```python
  dataset = tf.data.Dataset.range(3)
  dataset = dataset.map(tf.io.serialize_tensor)
  writer = tf.data.experimental.TFRecordWriter("/path/to/file.tfrecord")
  writer.write(dataset)
  ```

  To read back the elements, use `TFRecordDataset`.

  ```python
  dataset = tf.data.TFRecordDataset("/path/to/file.tfrecord")
  dataset = dataset.map(lambda x: tf.io.parse_tensor(x, tf.int64))
  ```

  To shard a `dataset` across multiple TFRecord files:

  ```python
  dataset = ... # dataset to be written

  def reduce_func(key, dataset):
    filename = tf.strings.join([PATH_PREFIX, tf.strings.as_string(key)])
    writer = tf.data.experimental.TFRecordWriter(filename)
    writer.write(dataset.map(lambda _, x: x))
    return tf.data.Dataset.from_tensors(filename)

  dataset = dataset.enumerate()
  dataset = dataset.apply(tf.data.experimental.group_by_window(
    lambda i, _: i % NUM_SHARDS, reduce_func, tf.int64.max
  ))

  # Iterate through the dataset to trigger data writing.
  for _ in dataset:
    pass
  ```
  """
  def __init__(self, filename, compression_type=...) -> None:
    """Initializes a `TFRecordWriter`.

    Args:
      filename: a string path indicating where to write the TFRecord data.
      compression_type: (Optional.) a string indicating what type of compression
        to use when writing the file. See `tf.io.TFRecordCompressionType` for
        what types of compression are available. Defaults to `None`.
    """
    ...
  
  def write(self, dataset): # -> object | Operation | None:
    """Writes a dataset to a TFRecord file.

    An operation that writes the content of the specified dataset to the file
    specified in the constructor.

    If the file exists, it will be overwritten.

    Args:
      dataset: a `tf.data.Dataset` whose elements are to be written to a file

    Returns:
      In graph mode, this returns an operation which when executed performs the
      write. In eager mode, the write is performed by the method itself and
      there is no return value.

    Raises
      TypeError: if `dataset` is not a `tf.data.Dataset`.
      TypeError: if the elements produced by the dataset are not scalar strings.
    """
    ...
  


