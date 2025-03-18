"""
This type stub file was generated by pyright.
"""

import enum
from tensorflow.python.feature_column import feature_column_lib as fc_lib
from tensorflow.python.tpu.feature_column import _TPUBaseEmbeddingColumn
from tensorflow.python.util.tf_export import tf_export

"""TPU Feature Column Library."""
_ALLOWED_DEVICES = ...
_TENSOR_CORE_MASK_KEY_SUFFIX = ...
class EmbeddingDevice(enum.Enum):
  CPU = ...
  TPU_TENSOR_CORE = ...
  TPU_EMBEDDING_CORE = ...


@tf_export(v1=['tpu.experimental.embedding_column'])
def embedding_column_v2(categorical_column, dimension, combiner=..., initializer=..., max_sequence_length=..., learning_rate_fn=..., embedding_lookup_device=..., tensor_core_shape=..., use_safe_embedding_lookup=...): # -> _TPUEmbeddingColumnV2 | _TPUDeviceSpecificEmbeddingColumnV2:
  """TPU version of `tf.compat.v1.feature_column.embedding_column`.

  Note that the interface for `tf.tpu.experimental.embedding_column` is
  different from that of `tf.compat.v1.feature_column.embedding_column`: The
  following arguments are NOT supported: `ckpt_to_load_from`,
  `tensor_name_in_ckpt`, `max_norm` and `trainable`.

  Use this function in place of `tf.compat.v1.feature_column.embedding_column`
  when you want to use the TPU to accelerate your embedding lookups via TPU
  embeddings.

  ```
  column = tf.feature_column.categorical_column_with_identity(...)
  tpu_column = tf.tpu.experimental.embedding_column(column, 10)
  ...
  def model_fn(features):
    dense_feature = tf.keras.layers.DenseFeature(tpu_column)
    embedded_feature = dense_feature(features)
    ...

  estimator = tf.estimator.tpu.TPUEstimator(
      model_fn=model_fn,
      ...
      embedding_config_spec=tf.estimator.tpu.experimental.EmbeddingConfigSpec(
        column=[tpu_column],
        ...))
  ```

  Args:
    categorical_column: A categorical column returned from
        `categorical_column_with_identity`, `weighted_categorical_column`,
        `categorical_column_with_vocabulary_file`,
        `categorical_column_with_vocabulary_list`,
        `sequence_categorical_column_with_identity`,
        `sequence_categorical_column_with_vocabulary_file`,
        `sequence_categorical_column_with_vocabulary_list`
    dimension: An integer specifying dimension of the embedding, must be > 0.
    combiner: A string specifying how to reduce if there are multiple entries
      in a single row for a non-sequence column. For more information, see
      `tf.feature_column.embedding_column`.
    initializer: A variable initializer function to be used in embedding
      variable initialization. If not specified, defaults to
      `tf.compat.v1.truncated_normal_initializer` with mean `0.0` and
      standard deviation `1/sqrt(dimension)`.
    max_sequence_length: An non-negative integer specifying the max sequence
      length. Any sequence shorter then this will be padded with 0 embeddings
      and any sequence longer will be truncated. This must be positive for
      sequence features and 0 for non-sequence features.
    learning_rate_fn: A function that takes global step and returns learning
      rate for the embedding table. If you intend to use the same learning rate
      for multiple embedding tables, please ensure that you pass the exact same
      python function to all calls of embedding_column, otherwise performence
      may suffer.
    embedding_lookup_device: The device on which to run the embedding lookup.
      Valid options are "cpu", "tpu_tensor_core", and "tpu_embedding_core".
      If specifying "tpu_tensor_core", a tensor_core_shape must be supplied.
      If not specified, the default behavior is embedding lookup on
      "tpu_embedding_core" for training and "cpu" for inference.
      Valid options for training : ["tpu_embedding_core", "tpu_tensor_core"]
      Valid options for serving :  ["cpu", "tpu_tensor_core"]
      For training, tpu_embedding_core is good for large embedding vocab (>1M),
      otherwise, tpu_tensor_core is often sufficient.
      For serving, doing embedding lookup on tpu_tensor_core during serving is
      a way to reduce host cpu usage in cases where that is a bottleneck.
    tensor_core_shape: If supplied, a list of integers which specifies
      the intended dense shape to run embedding lookup for this feature on
      TensorCore. The batch dimension can be left None or -1 to indicate
      a dynamic shape. Only rank 2 shapes currently supported.
    use_safe_embedding_lookup: If true, uses safe_embedding_lookup_sparse
      instead of embedding_lookup_sparse. safe_embedding_lookup_sparse ensures
      there are no empty rows and all weights and ids are positive at the
      expense of extra compute cost. This only applies to rank 2 (NxM) shaped
      input tensors. Defaults to true, consider turning off if the above checks
      are not needed. Note that having empty rows will not trigger any error
      though the output result might be 0 or omitted.

  Returns:
    A  `_TPUEmbeddingColumnV2`.

  Raises:
    ValueError: if `dimension` not > 0.
    ValueError: if `initializer` is specified but not callable.
  """
  ...

@tf_export(v1=['tpu.experimental.shared_embedding_columns'])
def shared_embedding_columns_v2(categorical_columns, dimension, combiner=..., initializer=..., shared_embedding_collection_name=..., max_sequence_lengths=..., learning_rate_fn=..., embedding_lookup_device=..., tensor_core_shape=..., use_safe_embedding_lookup=...): # -> list[Any]:
  """TPU version of `tf.compat.v1.feature_column.shared_embedding_columns`.

  Note that the interface for `tf.tpu.experimental.shared_embedding_columns` is
  different from that of `tf.compat.v1.feature_column.shared_embedding_columns`:
  The following arguments are NOT supported: `ckpt_to_load_from`,
  `tensor_name_in_ckpt`, `max_norm` and `trainable`.

  Use this function in place of
  tf.compat.v1.feature_column.shared_embedding_columns` when you want to use the
  TPU to accelerate your embedding lookups via TPU embeddings.

  ```
  column_a = tf.feature_column.categorical_column_with_identity(...)
  column_b = tf.feature_column.categorical_column_with_identity(...)
  tpu_columns = tf.tpu.experimental.shared_embedding_columns(
      [column_a, column_b], 10)
  ...
  def model_fn(features):
    dense_feature = tf.keras.layers.DenseFeature(tpu_columns)
    embedded_feature = dense_feature(features)
    ...

  estimator = tf.estimator.tpu.TPUEstimator(
      model_fn=model_fn,
      ...
      embedding_config_spec=tf.estimator.tpu.experimental.EmbeddingConfigSpec(
          column=tpu_columns,
          ...))
  ```

  Args:
    categorical_columns: A list of categorical columns returned from
      `categorical_column_with_identity`, `weighted_categorical_column`,
      `categorical_column_with_vocabulary_file`,
      `categorical_column_with_vocabulary_list`,
      `sequence_categorical_column_with_identity`,
      `sequence_categorical_column_with_vocabulary_file`,
      `sequence_categorical_column_with_vocabulary_list`
    dimension: An integer specifying dimension of the embedding, must be > 0.
    combiner: A string specifying how to reduce if there are multiple entries in
      a single row for a non-sequence column. For more information, see
      `tf.feature_column.embedding_column`.
    initializer: A variable initializer function to be used in embedding
      variable initialization. If not specified, defaults to
      `tf.truncated_normal_initializer` with mean `0.0` and standard deviation
      `1/sqrt(dimension)`.
    shared_embedding_collection_name: Optional name of the collection where
      shared embedding weights are added. If not given, a reasonable name will
      be chosen based on the names of `categorical_columns`. This is also used
      in `variable_scope` when creating shared embedding weights.
    max_sequence_lengths: An list of non-negative integers, either None or empty
      or the same length as the argument categorical_columns. Entries
      corresponding to non-sequence columns must be 0 and entries corresponding
      to sequence columns specify the max sequence length for the column. Any
      sequence shorter then this will be padded with 0 embeddings and any
      sequence longer will be truncated.
    learning_rate_fn: A function that takes global step and returns learning
      rate for the embedding table. If you intend to use the same learning rate
      for multiple embedding tables, please ensure that you pass the exact same
      python function to all calls of shared_embedding_columns, otherwise
      performence may suffer.
    embedding_lookup_device: The device on which to run the embedding lookup.
      Valid options are "cpu", "tpu_tensor_core", and "tpu_embedding_core". If
      specifying "tpu_tensor_core", a tensor_core_shape must be supplied.
      Defaults to "cpu". If not specified, the default behavior is embedding
      lookup on "tpu_embedding_core" for training and "cpu" for inference.
      Valid options for training : ["tpu_embedding_core", "tpu_tensor_core"]
      Valid options for serving :  ["cpu", "tpu_tensor_core"]
      For training, tpu_embedding_core is good for large embedding vocab (>1M),
      otherwise, tpu_tensor_core is often sufficient.
      For serving, doing embedding lookup on tpu_tensor_core during serving is
      a way to reduce host cpu usage in cases where that is a bottleneck.
    tensor_core_shape: If supplied, a list of integers which specifies the
      intended dense shape to run embedding lookup for this feature on
      TensorCore. The batch dimension can be left None or -1 to indicate a
      dynamic shape. Only rank 2 shapes currently supported.
    use_safe_embedding_lookup: If true, uses safe_embedding_lookup_sparse
      instead of embedding_lookup_sparse. safe_embedding_lookup_sparse ensures
      there are no empty rows and all weights and ids are positive at the
      expense of extra compute cost. This only applies to rank 2 (NxM) shaped
      input tensors. Defaults to true, consider turning off if the above checks
      are not needed. Note that having empty rows will not trigger any error
      though the output result might be 0 or omitted.

  Returns:
    A  list of `_TPUSharedEmbeddingColumnV2`.

  Raises:
    ValueError: if `dimension` not > 0.
    ValueError: if `initializer` is specified but not callable.
    ValueError: if `max_sequence_lengths` is specified and not the same length
      as `categorical_columns`.
    ValueError: if `max_sequence_lengths` is positive for a non sequence column
      or 0 for a sequence column.
  """
  ...

class _TPUEmbeddingColumnV2(_TPUBaseEmbeddingColumn, fc_lib.EmbeddingColumn):
  """Core Embedding Column."""
  def __new__(cls, categorical_column, dimension, combiner=..., initializer=..., max_sequence_length=..., learning_rate_fn=..., use_safe_embedding_lookup=..., bypass_scope_validation=...): # -> Self:
    ...
  
  def __getnewargs__(self): # -> tuple[Any, Any, Any, Any, int, Any | None, Any, bool]:
    ...
  
  def __deepcopy__(self, memo): # -> _TPUEmbeddingColumnV2:
    ...
  
  def __init__(self, categorical_column, dimension, combiner=..., initializer=..., max_sequence_length=..., learning_rate_fn=..., use_safe_embedding_lookup=..., bypass_scope_validation=...) -> None:
    ...
  
  def get_combiner(self):
    ...
  
  def get_embedding_table_size(self): # -> tuple[Any, Any]:
    """Returns num_ids and width."""
    ...
  
  def get_feature_key_name(self):
    """get_feature_key_name."""
    ...
  
  def get_weight_key_name(self): # -> None:
    """get_weight_key_name."""
    ...
  
  def get_embedding_var_name(self):
    """get_embedding_var_name."""
    ...
  
  def get_initializer(self):
    ...
  
  def is_categorical_column_weighted(self): # -> bool:
    """Check if the categorical column of the embedding column is weighted."""
    ...
  
  def create_state(self, state_manager): # -> None:
    ...
  
  def get_dense_tensor(self, transformation_cache, state_manager): # -> Any | Tensor | SparseTensor | IndexedSlices | SymbolicTensor:
    ...
  
  def get_sequence_dense_tensor(self, transformation_cache, state_manager): # -> Any | TensorSequenceLengthPair:
    ...
  


class _TPUSharedEmbeddingColumnV2(_TPUBaseEmbeddingColumn, fc_lib.SharedEmbeddingColumn):
  """Core Shared Embedding Column."""
  def __new__(cls, categorical_column, shared_embedding_column_creator, combiner=..., initializer=..., shared_embedding_collection_name=..., max_sequence_length=..., learning_rate_fn=..., use_safe_embedding_lookup=...): # -> Self:
    ...
  
  def __getnewargs__(self): # -> tuple[Any, Any, Any, Any | None, Any | None, int, Any | None]:
    ...
  
  def __deepcopy__(self, memo): # -> _TPUSharedEmbeddingColumnV2:
    ...
  
  def __init__(self, categorical_column, shared_embedding_column_creator, combiner=..., initializer=..., shared_embedding_collection_name=..., max_sequence_length=..., learning_rate_fn=..., use_safe_embedding_lookup=...) -> None:
    ...
  
  def get_combiner(self):
    ...
  
  def get_embedding_table_size(self): # -> tuple[Any, Any]:
    """Returns num_ids and width."""
    ...
  
  def get_feature_key_name(self):
    """get_feature_key_name."""
    ...
  
  def get_weight_key_name(self): # -> None:
    """get_weight_key_name."""
    ...
  
  def get_embedding_var_name(self): # -> None:
    """get_embedding_var_name."""
    ...
  
  def get_initializer(self): # -> None:
    ...
  
  def is_categorical_column_weighted(self): # -> bool:
    """Check if the categorical column of the embedding column is weighted."""
    ...
  
  def get_sequence_dense_tensor(self, transformation_cache, state_manager): # -> Any | TensorSequenceLengthPair:
    ...
  


def split_sequence_columns_v2(feature_columns): # -> tuple[list[Any], list[Any]]:
  """Split a list of _TPUEmbeddingColumn into sequence and non-sequence columns.

  For use in a TPUEstimator model_fn function. E.g.

  def model_fn(features):
    sequence_columns, feature_columns = (
        tf.tpu.feature_column.split_sequence_columns(feature_columns))
    input = tf.feature_column.input_layer(
        features=features, feature_columns=feature_columns)
    sequence_features, sequence_lengths = (
        tf.contrib.feature_column.sequence_input_layer(
            features=features, feature_columns=sequence_columns))

  Args:
    feature_columns: A list of _TPUEmbeddingColumns to split.

  Returns:
    Two lists of _TPUEmbeddingColumns, the first is the sequence columns and the
    second is the non-sequence columns.
  """
  ...

def sparse_embedding_aggregate_slice(params, values_and_values_mask, combiner=..., name=...):
  """Uses XLA's dynamic slice operations to perform embedding lookups.

  From third_party/cloud_tpu/models/movielens/tpu_embedding.py

  Args:
    params: Tensor of embedding table. Rank 2 (table_size x embedding dim)
    values_and_values_mask: is a two-tuple that contains: values - Tensor of
      embedding indices. Rank 2 (batch x n_indices) values_mask - Tensor of mask
      / weights. Rank 2 (batch x n_indices)
    combiner: The combiner to use for the embedding lookup. Currently supports
      'sum' and 'mean'.
    name: Optional name scope for created ops

  Returns:
    Rank 2 tensor of aggregated (per batch element) embedding vectors.

  Raises:
    ValueError: Combiner is not supported.
  """
  ...

def pad_sparse_embedding_lookup_indices(sparse_indices, padded_size): # -> tuple[Any, Any]:
  """Creates statically-sized Tensors containing indices and weights.

  From third_party/cloud_tpu/models/movielens/tpu_embedding.py

  Also computes sparse_indices.values % embedding_table_size, for equivalent
  functionality to sparse_column_with_integerized_feature. The returned
  padded weight Tensor also doubles as a mask indicating which values in
  the returned padded indices Tensor are indices versus padded zeros.

  Args:
    sparse_indices: SparseTensor of embedding lookup indices.
    padded_size: Number of columns of the returned Tensors. Indices which fall
      out of bounds will be truncated to the padded size.

  Returns:
    (sparse_indices.values padded to the specified size,
     a mask the same size as the returned padded values in which 0s
     indicate padded locations and 1s (or values from sparse_weights)
     indicate actual values)
  """
  ...

class _TPUDeviceSpecificEmbeddingColumnV2(_TPUEmbeddingColumnV2):
  """TPUEmbeddingColumn which allows serving on TensorCore."""
  def __new__(cls, *args, **kwargs): # -> Self:
    ...
  
  def __init__(self, *args, **kwargs) -> None:
    ...
  
  def __deepcopy__(self, memo): # -> _TPUDeviceSpecificEmbeddingColumnV2:
    ...
  
  def create_state(self, state_manager): # -> None:
    ...
  
  def get_dense_tensor(self, transformation_cache, state_manager): # -> Any | Tensor | SparseTensor | IndexedSlices | SymbolicTensor:
    """Private method that follows get_dense_tensor."""
    ...
  


class _TPUSharedDeviceSpecificEmbeddingColumnV2(_TPUSharedEmbeddingColumnV2):
  """TPUSharedEmbeddingColumnV2 which allows serving on TensorCore."""
  def __new__(cls, *args, **kwargs): # -> Self:
    ...
  
  def __init__(self, *args, **kwargs) -> None:
    ...
  
  def __deepcopy__(self, memo): # -> _TPUSharedDeviceSpecificEmbeddingColumnV2:
    ...
  


