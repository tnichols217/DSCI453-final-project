"""
This type stub file was generated by pyright.
"""

from tensorflow.security.fuzzing.py import annotation_types as _atypes
from typing import Any, TypeVar
from typing_extensions import Annotated

"""Python wrappers around TensorFlow ops.

This file is MACHINE GENERATED! Do not edit.
"""
TV_NonDeterministicInts_dtype = TypeVar("TV_NonDeterministicInts_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_NonDeterministicInts_shape_dtype = TypeVar("TV_NonDeterministicInts_shape_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
def non_deterministic_ints(shape: Annotated[Any, TV_NonDeterministicInts_shape_dtype], dtype: TV_NonDeterministicInts_dtype = ..., name=...) -> Annotated[Any, TV_NonDeterministicInts_dtype]:
  r"""Non-deterministically generates some integers.

  This op may use some OS-provided source of non-determinism (e.g. an RNG), so each execution will give different results.

  Args:
    shape: A `Tensor`. The shape of the output tensor.
    dtype: An optional `tf.DType`. Defaults to `tf.int64`.
      The type of the output.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
  """
  ...

NonDeterministicInts = ...
def non_deterministic_ints_eager_fallback(shape: Annotated[Any, TV_NonDeterministicInts_shape_dtype], dtype: TV_NonDeterministicInts_dtype, name, ctx) -> Annotated[Any, TV_NonDeterministicInts_dtype]:
  ...

def rng_read_and_skip(resource: Annotated[Any, _atypes.Resource], alg: Annotated[Any, _atypes.Int32], delta: Annotated[Any, _atypes.UInt64], name=...) -> Annotated[Any, _atypes.Int64]:
  r"""Advance the counter of a counter-based RNG.

  The state of the RNG after
  `rng_read_and_skip(n)` will be the same as that after `uniform([n])`
  (or any other distribution). The actual increment added to the
  counter is an unspecified implementation choice.

  In the case that the input algorithm is RNG_ALG_AUTO_SELECT, the counter in the state needs to be of size int64[2], the current maximal counter size among algorithms. In this case, this op will manage the counter as if it is an 128-bit integer with layout [lower_64bits, higher_64bits]. If an algorithm needs less than 128 bits for the counter, it should use the left portion of the int64[2]. In this way, the int64[2] is compatible with all current RNG algorithms (Philox, ThreeFry and xla::RandomAlgorithm::RNG_DEFAULT). Downstream RNG ops can thus use this counter with any RNG algorithm.

  Args:
    resource: A `Tensor` of type `resource`.
      The handle of the resource variable that stores the state of the RNG. The state consists of the counter followed by the key.
    alg: A `Tensor` of type `int32`. The RNG algorithm.
    delta: A `Tensor` of type `uint64`. The amount of advancement.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int64`.
  """
  ...

RngReadAndSkip = ...
def rng_read_and_skip_eager_fallback(resource: Annotated[Any, _atypes.Resource], alg: Annotated[Any, _atypes.Int32], delta: Annotated[Any, _atypes.UInt64], name, ctx) -> Annotated[Any, _atypes.Int64]:
  ...

def rng_skip(resource: Annotated[Any, _atypes.Resource], algorithm: Annotated[Any, _atypes.Int64], delta: Annotated[Any, _atypes.Int64], name=...): # -> object | Operation | None:
  r"""Advance the counter of a counter-based RNG.

  The state of the RNG after
  `rng_skip(n)` will be the same as that after `stateful_uniform([n])`
  (or any other distribution). The actual increment added to the
  counter is an unspecified implementation detail.

  Args:
    resource: A `Tensor` of type `resource`.
      The handle of the resource variable that stores the state of the RNG.
    algorithm: A `Tensor` of type `int64`. The RNG algorithm.
    delta: A `Tensor` of type `int64`. The amount of advancement.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  ...

RngSkip = ...
def rng_skip_eager_fallback(resource: Annotated[Any, _atypes.Resource], algorithm: Annotated[Any, _atypes.Int64], delta: Annotated[Any, _atypes.Int64], name, ctx): # -> None:
  ...

TV_StatefulRandomBinomial_S = TypeVar("TV_StatefulRandomBinomial_S", _atypes.Int32, _atypes.Int64)
TV_StatefulRandomBinomial_T = TypeVar("TV_StatefulRandomBinomial_T", _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int32, _atypes.Int64)
TV_StatefulRandomBinomial_dtype = TypeVar("TV_StatefulRandomBinomial_dtype", _atypes.Float32, _atypes.Float64, _atypes.Half, _atypes.Int32, _atypes.Int64)
def stateful_random_binomial(resource: Annotated[Any, _atypes.Resource], algorithm: Annotated[Any, _atypes.Int64], shape: Annotated[Any, TV_StatefulRandomBinomial_S], counts: Annotated[Any, TV_StatefulRandomBinomial_T], probs: Annotated[Any, TV_StatefulRandomBinomial_T], dtype: TV_StatefulRandomBinomial_dtype = ..., name=...) -> Annotated[Any, TV_StatefulRandomBinomial_dtype]:
  r"""TODO: add doc.

  Args:
    resource: A `Tensor` of type `resource`.
    algorithm: A `Tensor` of type `int64`.
    shape: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    counts: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `int32`, `int64`.
    probs: A `Tensor`. Must have the same type as `counts`.
    dtype: An optional `tf.DType` from: `tf.half, tf.float32, tf.float64, tf.int32, tf.int64`. Defaults to `tf.int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
  """
  ...

StatefulRandomBinomial = ...
def stateful_random_binomial_eager_fallback(resource: Annotated[Any, _atypes.Resource], algorithm: Annotated[Any, _atypes.Int64], shape: Annotated[Any, TV_StatefulRandomBinomial_S], counts: Annotated[Any, TV_StatefulRandomBinomial_T], probs: Annotated[Any, TV_StatefulRandomBinomial_T], dtype: TV_StatefulRandomBinomial_dtype, name, ctx) -> Annotated[Any, TV_StatefulRandomBinomial_dtype]:
  ...

TV_StatefulStandardNormal_dtype = TypeVar("TV_StatefulStandardNormal_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_StatefulStandardNormal_shape_dtype = TypeVar("TV_StatefulStandardNormal_shape_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
def stateful_standard_normal(resource: Annotated[Any, _atypes.Resource], shape: Annotated[Any, TV_StatefulStandardNormal_shape_dtype], dtype: TV_StatefulStandardNormal_dtype = ..., name=...) -> Annotated[Any, TV_StatefulStandardNormal_dtype]:
  r"""Outputs random values from a normal distribution. This op is deprecated in favor of op 'StatefulStandardNormalV2'

  The generated values will have mean 0 and standard deviation 1.

  Args:
    resource: A `Tensor` of type `resource`.
      The handle of the resource variable that stores the state of the RNG.
    shape: A `Tensor`. The shape of the output tensor.
    dtype: An optional `tf.DType`. Defaults to `tf.float32`.
      The type of the output.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
  """
  ...

StatefulStandardNormal = ...
def stateful_standard_normal_eager_fallback(resource: Annotated[Any, _atypes.Resource], shape: Annotated[Any, TV_StatefulStandardNormal_shape_dtype], dtype: TV_StatefulStandardNormal_dtype, name, ctx) -> Annotated[Any, TV_StatefulStandardNormal_dtype]:
  ...

TV_StatefulStandardNormalV2_dtype = TypeVar("TV_StatefulStandardNormalV2_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_StatefulStandardNormalV2_shape_dtype = TypeVar("TV_StatefulStandardNormalV2_shape_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
def stateful_standard_normal_v2(resource: Annotated[Any, _atypes.Resource], algorithm: Annotated[Any, _atypes.Int64], shape: Annotated[Any, TV_StatefulStandardNormalV2_shape_dtype], dtype: TV_StatefulStandardNormalV2_dtype = ..., name=...) -> Annotated[Any, TV_StatefulStandardNormalV2_dtype]:
  r"""Outputs random values from a normal distribution.

  The generated values will have mean 0 and standard deviation 1.

  Args:
    resource: A `Tensor` of type `resource`.
      The handle of the resource variable that stores the state of the RNG.
    algorithm: A `Tensor` of type `int64`. The RNG algorithm.
    shape: A `Tensor`. The shape of the output tensor.
    dtype: An optional `tf.DType`. Defaults to `tf.float32`.
      The type of the output.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
  """
  ...

StatefulStandardNormalV2 = ...
def stateful_standard_normal_v2_eager_fallback(resource: Annotated[Any, _atypes.Resource], algorithm: Annotated[Any, _atypes.Int64], shape: Annotated[Any, TV_StatefulStandardNormalV2_shape_dtype], dtype: TV_StatefulStandardNormalV2_dtype, name, ctx) -> Annotated[Any, TV_StatefulStandardNormalV2_dtype]:
  ...

TV_StatefulTruncatedNormal_dtype = TypeVar("TV_StatefulTruncatedNormal_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_StatefulTruncatedNormal_shape_dtype = TypeVar("TV_StatefulTruncatedNormal_shape_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
def stateful_truncated_normal(resource: Annotated[Any, _atypes.Resource], algorithm: Annotated[Any, _atypes.Int64], shape: Annotated[Any, TV_StatefulTruncatedNormal_shape_dtype], dtype: TV_StatefulTruncatedNormal_dtype = ..., name=...) -> Annotated[Any, TV_StatefulTruncatedNormal_dtype]:
  r"""Outputs random values from a truncated normal distribution.

  The generated values follow a normal distribution with mean 0 and standard
  deviation 1, except that values whose magnitude is more than 2 standard
  deviations from the mean are dropped and re-picked.

  Args:
    resource: A `Tensor` of type `resource`.
      The handle of the resource variable that stores the state of the RNG.
    algorithm: A `Tensor` of type `int64`. The RNG algorithm.
    shape: A `Tensor`. The shape of the output tensor.
    dtype: An optional `tf.DType`. Defaults to `tf.float32`.
      The type of the output.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
  """
  ...

StatefulTruncatedNormal = ...
def stateful_truncated_normal_eager_fallback(resource: Annotated[Any, _atypes.Resource], algorithm: Annotated[Any, _atypes.Int64], shape: Annotated[Any, TV_StatefulTruncatedNormal_shape_dtype], dtype: TV_StatefulTruncatedNormal_dtype, name, ctx) -> Annotated[Any, TV_StatefulTruncatedNormal_dtype]:
  ...

TV_StatefulUniform_dtype = TypeVar("TV_StatefulUniform_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_StatefulUniform_shape_dtype = TypeVar("TV_StatefulUniform_shape_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
def stateful_uniform(resource: Annotated[Any, _atypes.Resource], algorithm: Annotated[Any, _atypes.Int64], shape: Annotated[Any, TV_StatefulUniform_shape_dtype], dtype: TV_StatefulUniform_dtype = ..., name=...) -> Annotated[Any, TV_StatefulUniform_dtype]:
  r"""Outputs random values from a uniform distribution.

  The generated values follow a uniform distribution in the range `[0, 1)`. The
  lower bound 0 is included in the range, while the upper bound 1 is excluded.

  Args:
    resource: A `Tensor` of type `resource`.
      The handle of the resource variable that stores the state of the RNG.
    algorithm: A `Tensor` of type `int64`. The RNG algorithm.
    shape: A `Tensor`. The shape of the output tensor.
    dtype: An optional `tf.DType`. Defaults to `tf.float32`.
      The type of the output.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
  """
  ...

StatefulUniform = ...
def stateful_uniform_eager_fallback(resource: Annotated[Any, _atypes.Resource], algorithm: Annotated[Any, _atypes.Int64], shape: Annotated[Any, TV_StatefulUniform_shape_dtype], dtype: TV_StatefulUniform_dtype, name, ctx) -> Annotated[Any, TV_StatefulUniform_dtype]:
  ...

TV_StatefulUniformFullInt_dtype = TypeVar("TV_StatefulUniformFullInt_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_StatefulUniformFullInt_shape_dtype = TypeVar("TV_StatefulUniformFullInt_shape_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
def stateful_uniform_full_int(resource: Annotated[Any, _atypes.Resource], algorithm: Annotated[Any, _atypes.Int64], shape: Annotated[Any, TV_StatefulUniformFullInt_shape_dtype], dtype: TV_StatefulUniformFullInt_dtype = ..., name=...) -> Annotated[Any, TV_StatefulUniformFullInt_dtype]:
  r"""Outputs random integers from a uniform distribution.

  The generated values are uniform integers covering the whole range of `dtype`.

  Args:
    resource: A `Tensor` of type `resource`.
      The handle of the resource variable that stores the state of the RNG.
    algorithm: A `Tensor` of type `int64`. The RNG algorithm.
    shape: A `Tensor`. The shape of the output tensor.
    dtype: An optional `tf.DType`. Defaults to `tf.uint64`.
      The type of the output.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
  """
  ...

StatefulUniformFullInt = ...
def stateful_uniform_full_int_eager_fallback(resource: Annotated[Any, _atypes.Resource], algorithm: Annotated[Any, _atypes.Int64], shape: Annotated[Any, TV_StatefulUniformFullInt_shape_dtype], dtype: TV_StatefulUniformFullInt_dtype, name, ctx) -> Annotated[Any, TV_StatefulUniformFullInt_dtype]:
  ...

TV_StatefulUniformInt_dtype = TypeVar("TV_StatefulUniformInt_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
TV_StatefulUniformInt_shape_dtype = TypeVar("TV_StatefulUniformInt_shape_dtype", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
def stateful_uniform_int(resource: Annotated[Any, _atypes.Resource], algorithm: Annotated[Any, _atypes.Int64], shape: Annotated[Any, TV_StatefulUniformInt_shape_dtype], minval: Annotated[Any, TV_StatefulUniformInt_dtype], maxval: Annotated[Any, TV_StatefulUniformInt_dtype], name=...) -> Annotated[Any, TV_StatefulUniformInt_dtype]:
  r"""Outputs random integers from a uniform distribution.

  The generated values are uniform integers in the range `[minval, maxval)`.
  The lower bound `minval` is included in the range, while the upper bound
  `maxval` is excluded.

  The random integers are slightly biased unless `maxval - minval` is an exact
  power of two.  The bias is small for values of `maxval - minval` significantly
  smaller than the range of the output (either `2^32` or `2^64`).

  Args:
    resource: A `Tensor` of type `resource`.
      The handle of the resource variable that stores the state of the RNG.
    algorithm: A `Tensor` of type `int64`. The RNG algorithm.
    shape: A `Tensor`. The shape of the output tensor.
    minval: A `Tensor`. Minimum value (inclusive, scalar).
    maxval: A `Tensor`. Must have the same type as `minval`.
      Maximum value (exclusive, scalar).
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `minval`.
  """
  ...

StatefulUniformInt = ...
def stateful_uniform_int_eager_fallback(resource: Annotated[Any, _atypes.Resource], algorithm: Annotated[Any, _atypes.Int64], shape: Annotated[Any, TV_StatefulUniformInt_shape_dtype], minval: Annotated[Any, TV_StatefulUniformInt_dtype], maxval: Annotated[Any, TV_StatefulUniformInt_dtype], name, ctx) -> Annotated[Any, TV_StatefulUniformInt_dtype]:
  ...

