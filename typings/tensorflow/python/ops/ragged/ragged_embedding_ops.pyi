"""
This type stub file was generated by pyright.
"""

from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import dispatch

"""Embedding operations."""
@dispatch.dispatch_for_api(embedding_ops.embedding_lookup)
def embedding_lookup(params, ids: ragged_tensor.Ragged, partition_strategy=..., name=..., validate_indices=..., max_norm=...):
  """Look up the ragged ids in a list of embedding tensors.

  Args:
    params: A tensor representing the complete embedding tensor having the shape
      [e1, ...eM]
    ragged_ids: A 'RaggedTensor' with type 'int32' or 'int64' containing the ids
      to be looked up in 'params' of shape [r0, ..rN]. Values must be in the
      range '[0, params.shape[0]]'.
    partition_strategy: A string specifying the partitioning strategy.
    max_norm: If not `None`, each embedding is clipped if its l2-norm is larger
      than this value.
    name: A name for the operation (optional)

  Returns:
    A ragged tensor of shape [r0, r1, ...rN, e1, ...eM].

  Raises:
    ValueError: When params is empty or the type of the ids is not int32 or
      int64.
  """
  ...

@dispatch.dispatch_for_api(embedding_ops.embedding_lookup_sparse)
def embedding_lookup_sparse(params, sp_ids: ragged_tensor.Ragged, sp_weights, partition_strategy=..., name=..., combiner=..., max_norm=..., allow_fast_lookup=...): # -> Tensor | SparseTensor | IndexedSlices | SymbolicTensor:
  """Looks up embeddings for the given ids and weights from a list of tensors.

  This op assumes that there is at least one id for each row in the dense tensor
  represented by sp_ids (i.e. there are no rows with empty features), and that
  all the indices of sp_ids are in canonical row-major order.

  `sp_ids` and `sp_weights` (if not None) are `RaggedTensor`s with rank of 2.
  Embeddings are always aggregated along the last dimension.

  It also assumes that all id values lie in the range [0, p0), where p0
  is the sum of the size of params along dimension 0.

  Args:
    params: A single tensor representing the complete embedding tensor, or a
      list tensors all of same shape except for the first dimension,
      representing sharded embedding tensors. Alternatively, a
      `PartitionedVariable`, created by partitioning along dimension 0. Each
      element must be appropriately sized for the given `partition_strategy`.
    sp_ids: `RaggedTensor` with rank 2. The rank is not verified for performance
      reasons.
    sparse_weights: `RaggedTensor` of same type and shape as `sparse_ids`,
      containing float / double weights corresponding to `sparse_ids`, or `None`
      if all weights are assumed to be 1.0.
    partition_strategy: A string specifying the partitioning strategy, relevant
      if `len(params) > 1`. Currently `"div"` and `"mod"` are supported. Default
      is `"mod"`. See `tf.nn.embedding_lookup` for more details.
    name: Optional name for the op.
    combiner: A string specifying the reduction op. Currently "mean", "sqrtn"
      and "sum" are supported. "sum" computes the weighted sum of the embedding
      results for each row. "mean" is the weighted sum divided by the total
      weight. "sqrtn" is the weighted sum divided by the square root of the sum
      of the squares of the weights. Defaults to `mean`.
    max_norm: If not `None`, each embedding is clipped if its l2-norm is larger
      than this value, before combining.
    allow_fast_lookup: An optional boolean specifying whether to allow
      simplified embedding lookups when `params` is a single tensor and
      `max_norm` is `None`. Setting this flag to `True` during training can
      cause the use of dense gradients with increased memory footprint.

  Returns:
    A dense tensor representing the combined embeddings for the
    sparse ids. For each row in the dense tensor represented by `sp_ids`, the op
    looks up the embeddings for all ids in that row, multiplies them by the
    corresponding weight, and combines these embeddings as specified.

    In other words, if

      `shape(combined params) = [p0, p1, ..., pm]`

    and

      `shape(sp_ids) = shape(sp_weights) = [d0, d1]`

    then

      `shape(output) = [d0, p1, ..., pm]`.

    For instance, if params is a 10x20 matrix, and sp_ids / sp_weights are

      ```python
      [0, 0]: id 1, weight 2.0
      [0, 1]: id 3, weight 0.5
      [1, 0]: id 0, weight 1.0
      [2, 3]: id 1, weight 3.0
      ```

    with `combiner`="mean", then the output will be a 3x20 matrix where

      ```python
      output[0, :] = (params[1, :] * 2.0 + params[3, :] * 0.5) / (2.0 + 0.5)
      output[1, :] = (params[0, :] * 1.0) / 1.0
      output[2, :] = (params[1, :] * 3.0) / 3.0
      ```

  Raises:
    TypeError: If `sp_weights` is neither `None` nor of the same type as
      `sp_ids`.
    ValueError: If `combiner` is not one of {"mean", "sqrtn", "sum"}.
  """
  ...

@dispatch.dispatch_for_api(embedding_ops.safe_embedding_lookup_sparse)
def safe_embedding_lookup_sparse(embedding_weights, sparse_ids: ragged_tensor.Ragged, sparse_weights=..., combiner=..., default_id=..., name=..., partition_strategy=..., max_norm=..., allow_fast_lookup=...): # -> Any | Tensor | SparseTensor | IndexedSlices | SymbolicTensor:
  """Lookup embedding results, accounting for invalid IDs and empty features.

  The partitioned embedding in `embedding_weights` must all be the same shape
  except for the first dimension. The first dimension is allowed to vary as the
  vocabulary size is not necessarily a multiple of `P`.  `embedding_weights`
  may be a `PartitionedVariable` as returned by using
  `tf.compat.v1.get_variable()` with a
  partitioner.

  Invalid IDs (< 0) are pruned from input IDs and weights, as well as any IDs
  with non-positive weight. For an entry with no features, the embedding vector
  for `default_id` is returned, or the 0-vector if `default_id` is not supplied.

  The ids and weights may be multi-dimensional `SparseTensor`s or
  `RaggedTensor`s with rank of 2. For `SpareTensor`s with left-aligned non-zero
  entries which can be described as `RaggedTensor`s, use of `RaggedTensor`s can
  yield higher performance. Embeddings are always aggregated along the last
  dimension.

  Args:
    embedding_weights: A single tensor representing the complete embedding
      tensor, or a list tensors all of same shape except for the first
      dimension, representing sharded embedding tensors. Alternatively, a
      `PartitionedVariable`, created by partitioning along dimension 0. Each
      element must be appropriately sized for the given `partition_strategy`.
    sp_ids: `RaggedTensor` with rank 2. The rank is not verified for performance
      reasons.
    sparse_weights: `RaggedTensor` of same type and shape as `sparse_ids`,
      containing float weights corresponding to `sparse_ids`, or `None` if all
      weights are assumed to be 1.0.
    combiner: A string specifying how to combine embedding results for each
      entry. Currently "mean", "sqrtn" and "sum" are supported, with "mean" the
      default.
    default_id: The id to use for an entry with no features.
    name: A name for this operation (optional).
    partition_strategy: A string specifying the partitioning strategy. Currently
      `"div"` and `"mod"` are supported. Default is `"div"`.
    max_norm: If not `None`, all embeddings are l2-normalized to max_norm before
      combining.
    allow_fast_lookup: An optional boolean specifying whether to allow
      simplified embedding lookups when `params` is a single tensor and
      `max_norm` is `None`. Setting this flag to `True` during training can
      cause the use of dense gradients with increased memory footprint.

  Returns:
    A dense tensor representing the combined embeddings for the
    sparse ids. For each row in the dense tensor represented by `sp_ids`, the op
    looks up the embeddings for all ids in that row, multiplies them by the
    corresponding weight, and combines these embeddings as specified.

    In other words, if

      `shape(combined embedding_weights) = [p0, p1, ..., pm]`

    and

      `shape(sparse_ids) = shape(sparse_weights) = [d0, d1, ..., dn]`

    then

      `shape(output) = [d0, d1, ... dn-1, p1, ..., pm]`.

    For instance, if params is a 10x20 matrix, and sp_ids / sp_weights are

      ```python
      [0, 0]: id 1, weight 2.0
      [0, 1]: id 3, weight 0.5
      [1, 0]: id -1, weight 1.0
      [2, 3]: id 1, weight 3.0
      ```

    `default_id` is 0.

    with `combiner`="mean", then the output will be a 3x20 matrix where

      ```python
      output[0, :] = (params[1, :] * 2.0 + params[3, :] * 0.5) / (2.0 + 0.5)
      output[1, :] = (params[0, :] * 1.0) / 1.0
      output[2, :] = (params[1, :] * 3.0) / 3.0
      ```

  Raises:
    ValueError: if `embedding_weights` is empty.
  """
  ...

