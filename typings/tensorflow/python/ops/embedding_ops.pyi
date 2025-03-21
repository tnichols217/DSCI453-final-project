"""
This type stub file was generated by pyright.
"""

from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export

"""Operations for embeddings."""
@tf_export(v1=["nn.embedding_lookup"])
@dispatch.add_dispatch_support
def embedding_lookup(params, ids, partition_strategy=..., name=..., validate_indices=..., max_norm=...): # -> defaultdict[Any, Any] | Any | list[Any] | object | IndexedSlices | None:
  """Looks up embeddings for the given `ids` from a list of tensors.

  This function is used to perform parallel lookups on the list of tensors in
  `params`.  It is a generalization of `tf.gather`, where `params` is
  interpreted as a partitioning of a large embedding tensor.  `params` may be
  a `PartitionedVariable` as returned by using `tf.compat.v1.get_variable()`
  with a partitioner.

  If `len(params) > 1`, each element `id` of `ids` is partitioned between
  the elements of `params` according to the `partition_strategy`.
  In all strategies, if the id space does not evenly divide the number of
  partitions, each of the first `(max_id + 1) % len(params)` partitions will
  be assigned one more id.

  If `partition_strategy` is `"mod"`, we assign each id to partition
  `p = id % len(params)`. For instance,
  13 ids are split across 5 partitions as:
  `[[0, 5, 10], [1, 6, 11], [2, 7, 12], [3, 8], [4, 9]]`

  If `partition_strategy` is `"div"`, we assign ids to partitions in a
  contiguous manner. In this case, 13 ids are split across 5 partitions as:
  `[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10], [11, 12]]`

  If the input ids are ragged tensors, partition variables are not supported and
  the partition strategy and the max_norm are ignored.
  The results of the lookup are concatenated into a dense
  tensor. The returned tensor has shape `shape(ids) + shape(params)[1:]`.

  Args:
    params: A single tensor representing the complete embedding tensor, or a
      list of P tensors all of same shape except for the first dimension,
      representing sharded embedding tensors.  Alternatively, a
      `PartitionedVariable`, created by partitioning along dimension 0. Each
      element must be appropriately sized for the given `partition_strategy`.
    ids: A `Tensor` or a 'RaggedTensor' with type `int32` or `int64` containing
      the ids to be looked up in `params`.
      Caution: Out-of-bounds indices will result in undefined behavior, which
        will differ between devices and backends.
    partition_strategy: A string specifying the partitioning strategy, relevant
      if `len(params) > 1`. Currently `"div"` and `"mod"` are supported. Default
      is `"mod"`.
    name: A name for the operation (optional).
    validate_indices: DEPRECATED. If this operation is assigned to CPU, values
      in `indices` are always validated to be within range.  If assigned to GPU,
      out-of-bound indices result in safe but unspecified behavior, which may
      include raising an error.
    max_norm: If not `None`, each embedding is clipped if its l2-norm is larger
      than this value.

  Returns:
    A `Tensor` or a 'RaggedTensor', depending on the input, with the same type
    as the tensors in `params`.

  Raises:
    ValueError: If `params` is empty.
  """
  ...

@tf_export("nn.embedding_lookup", v1=[])
@dispatch.add_dispatch_support
def embedding_lookup_v2(params, ids, max_norm=..., name=...): # -> defaultdict[Any, Any] | Any | list[Any] | object | IndexedSlices | None:
  """Looks up embeddings for the given `ids` from a list of tensors.

  This function is used to perform parallel lookups on the list of tensors in
  `params`.  It is a generalization of `tf.gather`, where `params` is
  interpreted as a partitioning of a large embedding tensor.

  If `len(params) > 1`, each element `id` of `ids` is partitioned between the
  elements of `params` according to the "div" partition strategy, which means we
  assign ids to partitions in a contiguous manner. For instance, 13 ids are
  split across 5 partitions as:
  `[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10], [11, 12]]`.

  If the id space does not evenly divide the number of partitions, each of the
  first `(max_id + 1) % len(params)` partitions will be assigned one more id.

  The results of the lookup are concatenated into a dense
  tensor. The returned tensor has shape `shape(ids) + shape(params)[1:]`.

  Args:
    params: A single tensor representing the complete embedding tensor, or a
      list of tensors all of same shape except for the first dimension,
      representing sharded embedding tensors following "div" partition strategy.
    ids: A `Tensor` with type `int32` or `int64` containing the ids to be looked
      up in `params`.
    max_norm: If not `None`, each embedding is clipped if its l2-norm is larger
      than this value.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` with the same type as the tensors in `params`.

    For instance, if `params` is a 5x2 matrix:

    ```python
    [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
    ```

    or a list of matrices:

    ```python
    params[0]: [[1, 2], [3, 4]]
    params[1]: [[5, 6], [7, 8]]
    params[2]: [[9, 10]]
    ```

    and `ids` is:

    ```python
    [0, 3, 4]
    ```

    The output will be a 3x2 matrix:

    ```python
    [[1, 2], [7, 8], [9, 10]]
    ```

  Raises:
    ValueError: If `params` is empty.
  """
  ...

@tf_export(v1=["nn.embedding_lookup_sparse"])
@dispatch.add_dispatch_support
def embedding_lookup_sparse(params, sp_ids, sp_weights, partition_strategy=..., name=..., combiner=..., max_norm=..., allow_fast_lookup=...): # -> Tensor | SparseTensor | IndexedSlices | SymbolicTensor:
  """Looks up embeddings for the given ids and weights from a list of tensors.

  This op assumes that there is at least one id for each row in the dense tensor
  represented by sp_ids (i.e. there are no rows with empty features), and that
  all the indices of sp_ids are in canonical row-major order.

  `sp_ids` and `sp_weights` (if not None) are `SparseTensor`s or `RaggedTensor`s
  with rank of 2. For `SpareTensor`s with left-aligned non-zero entries which
  can be described as `RaggedTensor`s, use of `RaggedTensor`s can yield higher
  performance.

  It also assumes that all id values lie in the range [0, p0), where p0
  is the sum of the size of params along dimension 0.

  Args:
    params: A single tensor representing the complete embedding tensor, or a
      list tensors all of same shape except for the first dimension,
      representing sharded embedding tensors. Alternatively, a
      `PartitionedVariable`, created by partitioning along dimension 0. Each
      element must be appropriately sized for the given `partition_strategy`.
    sp_ids: N x M `SparseTensor` of int64 ids where N is typically batch size
      and M is arbitrary or a `RaggedTensor` with rank 2.
    sparse_weights: `SparseTensor` or `RaggedTensor` of same type and shape as
      `sparse_ids`, containing float / double weights corresponding to
      `sparse_ids`, or `None` if all weights are assumed to be 1.0.
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
    TypeError: If `sp_ids` is not a `SparseTensor` or `RaggedTensor`, or if
    `sp_weights` is neither `None` nor of the same type as `sp_ids`.
    ValueError: If `combiner` is not one of {"mean", "sqrtn", "sum"}.
  """
  ...

@tf_export("nn.embedding_lookup_sparse", v1=[])
@dispatch.add_dispatch_support
def embedding_lookup_sparse_v2(params, sp_ids, sp_weights, combiner=..., max_norm=..., name=..., allow_fast_lookup=...): # -> Tensor | SparseTensor | IndexedSlices | SymbolicTensor:
  """Looks up embeddings for the given ids and weights from a list of tensors.

  `params` is a dense tensor or a list of dense tensors, and `sp_ids` is a 2D
  `tf.SparseTensor` or `tf.RaggedTensor` indicating the indices of `params` to
  gather.

  This op is best described with an example. Suppose `params` is an embedding
  table of size `(4, 2)` and `sp_ids` has 3 rows. Since `sp_ids` is sparse or
  ragged, not every row has the same number of elements. The output has shape
  (3, 2). Each row of `sp_ids` is a list of indices, where each index selects a
  row of `params`. For a given row of `sp_ids`, the rows of `params` are
  gathered based on the indices in `sp_ids`, then combined by taking their sum
  or mean.

  >>> params = tf.constant([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=tf.float32)
  >>> sp_ids = tf.SparseTensor(indices=[[0, 0], [0, 1], [1, 0], [2, 0]],
  ...                          values=[0, 1, 3, 2], dense_shape=(3, 2))
  >>> tf.nn.embedding_lookup_sparse(params, sp_ids, sp_weights=None,
  ...                               combiner='sum').numpy()
  array([[4., 6.], [7., 8.], [5., 6.]], dtype=float32)

  In this example, `sp_ids` has 3 rows, so the output has 3 rows. Row 0 of
  `sp_ids` has values 0 and 1, so it selects rows 0 and 1 from `params`, which
  are `[1, 2]` and `[3, 4]`. The rows are summed since `combiner='sum'`,
  resulting in the output row of `[4, 6]`.

  Since row 1 and 2 of `sp_ids` only have one value each, they simply select the
  corresponding row from `params` as the output row. Row 1 has value `3` so
  it selects the `params` elements `[7, 8]` and row 2 has the value 2 so it
  selects the `params` elements `[5, 6]`.

  If `sparse_weights` is specified, it must have the same shape as `sp_ids`.
  `sparse_weights` is used to assign a weight to each slice of `params`. For
  example:

  >>> params = tf.constant([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=tf.float32)
  >>> sp_ids = tf.SparseTensor(indices=[[0, 0], [0, 1], [1, 0], [2, 0]],
  ...                          values=[0, 1, 3, 2], dense_shape=(3, 2))
  >>> sparse_weights = tf.SparseTensor(indices=[[0, 0], [0, 1], [1, 0], [2, 0]],
  ...                                  values=[0.1, 1.0, 0.5, 2.0],
  ...                                  dense_shape=(3, 2))
  >>> tf.nn.embedding_lookup_sparse(params, sp_ids, sp_weights=sparse_weights,
  ...                               combiner='sum').numpy()
  array([[3.1, 4.2], [3.5, 4.], [10., 12.]], dtype=float32)

  In general, `params` can have shape `(p0, ..., pn)` and `sp_ids` can have `M`
  rows, where each row can have any number of elements. The output has shape
  `(M, p1, ..., pn)`. Each slice of the output `output[i, ...]` is obtained as
  follows: The `combiner` argument is used to combine the values
  `params[sp_ids[i, j], ...] * sparse_weights[i, j]` for each `j` in `range(0,
  len(sp_ids[i]))`, e.g. by taking the sum or mean of the values.

  This op assumes that there is at least one id for each row in the dense tensor
  represented by sp_ids (i.e. there are no rows with empty features), and that
  all the indices of sp_ids are in canonical row-major order.

  `sp_ids` and `sp_weights` (if not None) are `SparseTensor`s or `RaggedTensor`s
  with rank of 2. For `SpareTensor`s with left-aligned non-zero entries which
  can be described as `RaggedTensor`s, use of `RaggedTensor`s can yield higher
  performance.

  This op assumes that all id values lie in the range [0, p0), where p0
  is `params.shape[0]`. If you want a version of this op that prunes id values
  less than 0, see `tf.nn.safe_embedding_lookup_sparse`

  If `len(params) > 1`, each element of `sp_ids` is partitioned between the
  elements of `params` according to the "div" partition strategy, which means we
  assign ids to partitions in a contiguous manner. For instance, 13 ids are
  split across 5 partitions as:
  `[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10], [11, 12]]`.

  If the id space does not evenly divide the number of partitions, each of the
  first `(max_id + 1) % len(params)` partitions will be assigned one more id.

  Args:
    params: A single tensor representing the complete embedding tensor, or a
      list of tensors all of same shape except for the first dimension,
      representing sharded embedding tensors following "div" partition strategy.
    sp_ids: N x M `SparseTensor` of int64 ids where N is typically batch size
      and M is arbitrary or a `RaggedTensor` with rank 2.
    sparse_weights: `SparseTensor` or `RaggedTensor` of same type and shape as
      `sparse_ids`, containing float / double weights corresponding to
      `sparse_ids`, or `None` if all weights are assumed to be 1.0.
    combiner: A string specifying the reduction op. Currently "mean", "sqrtn"
      and "sum" are supported. "sum" computes the weighted sum of the embedding
      results for each row. "mean" is the weighted sum divided by the total
      weight. "sqrtn" is the weighted sum divided by the square root of the sum
      of the squares of the weights. Defaults to `mean`.
    max_norm: If not `None`, each embedding is clipped if its l2-norm is larger
      than this value, before combining.
    name: Optional name for the op.
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
    TypeError: If `sp_ids` is not a `SparseTensor`, or if `sp_weights` is
      neither `None` nor `SparseTensor`.
    ValueError: If `combiner` is not one of {"mean", "sqrtn", "sum"}.
  """
  ...

@tf_export("nn.safe_embedding_lookup_sparse", v1=[])
@dispatch.add_dispatch_support
def safe_embedding_lookup_sparse_v2(embedding_weights, sparse_ids, sparse_weights=..., combiner=..., default_id=..., max_norm=..., name=..., allow_fast_lookup=...): # -> Any:
  """Lookup embedding results, accounting for invalid IDs and empty features.

  The partitioned embedding in `embedding_weights` must all be the same shape
  except for the first dimension. The first dimension is allowed to vary as the
  vocabulary size is not necessarily a multiple of num of shards.

  This is similar to `tf.nn.embedding_lookup_sparse`, except invalid IDs (< 0)
  are pruned from input IDs and weights, as well as any IDs with non-positive
  weight. For an entry with no features, the embedding vector for `default_id`
  is returned, or the 0-vector if `default_id` is not supplied. See
  `tf.nn.embedding_lookup_sparse` for more information on how sparse embedding
  lookups work in general.

  The ids and weights may be multi-dimensional `SparseTensor`s or
  `RaggedTensor`s with rank of 2. For `SpareTensor`s with left-aligned non-zero
  entries which can be described as `RaggedTensor`s, use of `RaggedTensor`s can
  yield higher performance.

  If `len(embedding_weights) > 1`, each element `id` of `ids` is partitioned
  between the elements of `embedding_weights` according to the "div" partition
  strategy, which means we assign ids to partitions in a contiguous manner. For
  instance, 13 ids are split across 5 partitions as:
  `[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10], [11, 12]]`.

  If the id space does not evenly divide the number of partitions, each of the
  first `(max_id + 1) % len(embedding_weights)` partitions will be assigned one
  more id.

  Args:
    embedding_weights: A single tensor representing the complete embedding
      tensor, or a list of tensors all of same shape except for the first
      dimension, representing sharded embedding tensors following "div"
      partition strategy.
    sparse_ids: `SparseTensor` of shape `[d_0, d_1, ..., d_n]` containing the
      ids, where `d_0` is typically batch size, or a `RaggedTensor` with rank 2.
    sparse_weights: `SparseTensor` or `RaggedTensor` of same type and shape as
      `sparse_ids`, containing float weights corresponding to `sparse_ids`, or
      `None` if all weights are assumed to be 1.0.
    combiner: A string specifying how to combine embedding results for each
      entry. Currently "mean", "sqrtn" and "sum" are supported, with "mean" the
      default.
    default_id: The id to use for an entry with no features. Defaults to
      0-vector.
    max_norm: If not `None`, all embeddings are l2-normalized to max_norm before
      combining.
    name: A name for this operation (optional).
    allow_fast_lookup: An optional boolean specifying whether to allow
      simplified embedding lookups when `params` is a single tensor and
      `max_norm` is `None`. Setting this flag to `True` during training can
      cause the use of dense gradients with increased memory footprint.

  Returns:
    A dense tensor representing the combined embeddings for the
    sparse ids. For each row in the dense tensor represented by `sparse_ids`,
    the op looks up the embeddings for all ids in that row, multiplies them by
    the corresponding weight, and combines these embeddings as specified.

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

@tf_export(v1=["nn.safe_embedding_lookup_sparse"])
@dispatch.add_dispatch_support
def safe_embedding_lookup_sparse(embedding_weights, sparse_ids, sparse_weights=..., combiner=..., default_id=..., name=..., partition_strategy=..., max_norm=..., allow_fast_lookup=...): # -> Any:
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
    sparse_ids: `SparseTensor` of shape `[d_0, d_1, ..., d_n]` containing the
      ids, where `d_0` is typically batch size, or a `RaggedTensor` with rank 2.
    sparse_weights: `SparseTensor` or `RaggedTensor` of same type and shape as
      `sparse_ids`, containing float weights corresponding to `sparse_ids`, or
      `None` if all weights are assumed to be 1.0.
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

def embedding_lookup_sparse_impl(params, segment_ids, sp_weights, ids, combiner, ignore_weights, max_norm, allow_fast_lookup, partition_strategy, name): # -> Tensor | SparseTensor | IndexedSlices | SymbolicTensor:
  """Implementation of sparse embedding aggregation."""
  ...

