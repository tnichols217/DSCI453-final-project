"""
This type stub file was generated by pyright.
"""

from tensorflow.security.fuzzing.py import annotation_types as _atypes
from typing import Any, TypeVar
from typing_extensions import Annotated

"""Python wrappers around TensorFlow ops.

This file is MACHINE GENERATED! Do not edit.
"""
_DenseCountSparseOutputOutput = ...
TV_DenseCountSparseOutput_T = TypeVar("TV_DenseCountSparseOutput_T", _atypes.Int32, _atypes.Int64)
TV_DenseCountSparseOutput_output_type = TypeVar("TV_DenseCountSparseOutput_output_type", _atypes.Float32, _atypes.Float64, _atypes.Int32, _atypes.Int64)
def dense_count_sparse_output(values: Annotated[Any, TV_DenseCountSparseOutput_T], weights: Annotated[Any, TV_DenseCountSparseOutput_output_type], binary_output: bool, minlength: int = ..., maxlength: int = ..., name=...): # -> DenseCountSparseOutput:
  r"""Performs sparse-output bin counting for a tf.tensor input.

    Counts the number of times each value occurs in the input.

  Args:
    values: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      Tensor containing data to count.
    weights: A `Tensor`. Must be one of the following types: `int32`, `int64`, `float32`, `float64`.
      A Tensor of the same shape as indices containing per-index weight values. May
      also be the empty tensor if no weights are used.
    binary_output: A `bool`.
      Whether to output the number of occurrences of each value or 1.
    minlength: An optional `int` that is `>= -1`. Defaults to `-1`.
      Minimum value to count. Can be set to -1 for no minimum.
    maxlength: An optional `int` that is `>= -1`. Defaults to `-1`.
      Maximum value to count. Can be set to -1 for no maximum.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output_indices, output_values, output_dense_shape).

    output_indices: A `Tensor` of type `int64`.
    output_values: A `Tensor`. Has the same type as `weights`.
    output_dense_shape: A `Tensor` of type `int64`.
  """
  ...

DenseCountSparseOutput = ...
def dense_count_sparse_output_eager_fallback(values: Annotated[Any, TV_DenseCountSparseOutput_T], weights: Annotated[Any, TV_DenseCountSparseOutput_output_type], binary_output: bool, minlength: int, maxlength: int, name, ctx): # -> DenseCountSparseOutput:
  ...

_RaggedCountSparseOutputOutput = ...
TV_RaggedCountSparseOutput_T = TypeVar("TV_RaggedCountSparseOutput_T", _atypes.Int32, _atypes.Int64)
TV_RaggedCountSparseOutput_output_type = TypeVar("TV_RaggedCountSparseOutput_output_type", _atypes.Float32, _atypes.Float64, _atypes.Int32, _atypes.Int64)
def ragged_count_sparse_output(splits: Annotated[Any, _atypes.Int64], values: Annotated[Any, TV_RaggedCountSparseOutput_T], weights: Annotated[Any, TV_RaggedCountSparseOutput_output_type], binary_output: bool, minlength: int = ..., maxlength: int = ..., name=...): # -> RaggedCountSparseOutput:
  r"""Performs sparse-output bin counting for a ragged tensor input.

    Counts the number of times each value occurs in the input.

  Args:
    splits: A `Tensor` of type `int64`.
      Tensor containing the row splits of the ragged tensor to count.
    values: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      Tensor containing values of the sparse tensor to count.
    weights: A `Tensor`. Must be one of the following types: `int32`, `int64`, `float32`, `float64`.
      A Tensor of the same shape as indices containing per-index weight values.
      May also be the empty tensor if no weights are used.
    binary_output: A `bool`.
      Whether to output the number of occurrences of each value or 1.
    minlength: An optional `int` that is `>= -1`. Defaults to `-1`.
      Minimum value to count. Can be set to -1 for no minimum.
    maxlength: An optional `int` that is `>= -1`. Defaults to `-1`.
      Maximum value to count. Can be set to -1 for no maximum.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output_indices, output_values, output_dense_shape).

    output_indices: A `Tensor` of type `int64`.
    output_values: A `Tensor`. Has the same type as `weights`.
    output_dense_shape: A `Tensor` of type `int64`.
  """
  ...

RaggedCountSparseOutput = ...
def ragged_count_sparse_output_eager_fallback(splits: Annotated[Any, _atypes.Int64], values: Annotated[Any, TV_RaggedCountSparseOutput_T], weights: Annotated[Any, TV_RaggedCountSparseOutput_output_type], binary_output: bool, minlength: int, maxlength: int, name, ctx): # -> RaggedCountSparseOutput:
  ...

_SparseCountSparseOutputOutput = ...
TV_SparseCountSparseOutput_T = TypeVar("TV_SparseCountSparseOutput_T", _atypes.Int32, _atypes.Int64)
TV_SparseCountSparseOutput_output_type = TypeVar("TV_SparseCountSparseOutput_output_type", _atypes.Float32, _atypes.Float64, _atypes.Int32, _atypes.Int64)
def sparse_count_sparse_output(indices: Annotated[Any, _atypes.Int64], values: Annotated[Any, TV_SparseCountSparseOutput_T], dense_shape: Annotated[Any, _atypes.Int64], weights: Annotated[Any, TV_SparseCountSparseOutput_output_type], binary_output: bool, minlength: int = ..., maxlength: int = ..., name=...): # -> SparseCountSparseOutput:
  r"""Performs sparse-output bin counting for a sparse tensor input.

    Counts the number of times each value occurs in the input.

  Args:
    indices: A `Tensor` of type `int64`.
      Tensor containing the indices of the sparse tensor to count.
    values: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      Tensor containing values of the sparse tensor to count.
    dense_shape: A `Tensor` of type `int64`.
      Tensor containing the dense shape of the sparse tensor to count.
    weights: A `Tensor`. Must be one of the following types: `int32`, `int64`, `float32`, `float64`.
      A Tensor of the same shape as indices containing per-index weight values.
      May also be the empty tensor if no weights are used.
    binary_output: A `bool`.
      Whether to output the number of occurrences of each value or 1.
    minlength: An optional `int` that is `>= -1`. Defaults to `-1`.
      Minimum value to count. Can be set to -1 for no minimum.
    maxlength: An optional `int` that is `>= -1`. Defaults to `-1`.
      Maximum value to count. Can be set to -1 for no maximum.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output_indices, output_values, output_dense_shape).

    output_indices: A `Tensor` of type `int64`.
    output_values: A `Tensor`. Has the same type as `weights`.
    output_dense_shape: A `Tensor` of type `int64`.
  """
  ...

SparseCountSparseOutput = ...
def sparse_count_sparse_output_eager_fallback(indices: Annotated[Any, _atypes.Int64], values: Annotated[Any, TV_SparseCountSparseOutput_T], dense_shape: Annotated[Any, _atypes.Int64], weights: Annotated[Any, TV_SparseCountSparseOutput_output_type], binary_output: bool, minlength: int, maxlength: int, name, ctx): # -> SparseCountSparseOutput:
  ...

