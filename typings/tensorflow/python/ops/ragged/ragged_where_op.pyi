"""
This type stub file was generated by pyright.
"""

import typing
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import dispatch

"""where operation for RaggedTensors."""
@dispatch.dispatch_for_api(array_ops.where_v2)
def where_v2(condition: ragged_tensor.RaggedOrDense, x: typing.Optional[ragged_tensor.RaggedOrDense] = ..., y: typing.Optional[ragged_tensor.RaggedOrDense] = ..., name=...): # -> Any | defaultdict[Any, Any] | list[Any] | object | None:
  """Return the elements where `condition` is `True`.

  : If both `x` and `y` are None: Retrieve indices of true elements.

    Returns the coordinates of true elements of `condition`. The coordinates
    are returned in a 2-D tensor with shape
    `[num_true_values, dim_size(condition)]`, where `result[i]` is the
    coordinates of the `i`th true value (in row-major order).

  : If both `x` and `y` are non-`None`: Multiplex between `x` and `y`.

    Choose an output shape  from the shapes of `condition`, `x`, and `y` that
    all three shapes are broadcastable to; and then use the broadcasted
    `condition` tensor as a mask that chooses whether the corredsponding element
    in the output should be taken from `x` (if `condition` is true) or `y` (if
    `condition` is false).

  >>> # Example: retrieve indices of true elements
  >>> tf.where(tf.ragged.constant([[True, False], [True]]))
  <tf.Tensor: shape=(2, 2), dtype=int64, numpy= array([[0, 0], [1, 0]])>

  >>> # Example: multiplex between `x` and `y`
  >>> tf.where(tf.ragged.constant([[True, False], [True, False, True]]),
  ...          tf.ragged.constant([['A', 'B'], ['C', 'D', 'E']]),
  ...          tf.ragged.constant([['a', 'b'], ['c', 'd', 'e']]))
  <tf.RaggedTensor [[b'A', b'b'], [b'C', b'd', b'E']]>

  Args:
    condition: A potentially ragged tensor of type `bool`
    x: A potentially ragged tensor (optional).
    y: A potentially ragged tensor (optional).  Must be specified if `x` is
      specified.  Must have the same rank and type as `x`.
    name: A name of the operation (optional).

  Returns:
    : If both `x` and `y` are `None`:
      A `Tensor` with shape `(num_true, rank(condition))`.
    : Otherwise:
      A potentially ragged tensor with the same type as `x` and `y`, and whose
      shape is broadcast-compatible with `x`, `y`, and `condition`.

  Raises:
    ValueError: When exactly one of `x` or `y` is non-`None`; or when
      `condition`, `x`, and `y` have incompatible shapes.
  """
  ...

@dispatch.dispatch_for_api(array_ops.where)
def where(condition: ragged_tensor.RaggedOrDense, x: typing.Optional[ragged_tensor.RaggedOrDense] = ..., y: typing.Optional[ragged_tensor.RaggedOrDense] = ..., name=...): # -> Any | defaultdict[Any, Any] | list[Any] | object | RaggedTensor | None:
  """Return the elements, either from `x` or `y`, depending on the `condition`.

  : If both `x` and `y` are `None`:
    Returns the coordinates of true elements of `condition`. The coordinates
    are returned in a 2-D tensor with shape
    `[num_true_values, dim_size(condition)]`, where `result[i]` is the
    coordinates of the `i`th true value (in row-major order).

  : If both `x` and `y` are non-`None`:
    Returns a tensor formed by selecting values from `x` where condition is
    true, and from `y` when condition is false.  In particular:

    : If `condition`, `x`, and `y` all have the same shape:

      * `result[i1...iN] = x[i1...iN]` if `condition[i1...iN]` is true.
      * `result[i1...iN] = y[i1...iN]` if `condition[i1...iN]` is false.

    : Otherwise:

      * `condition` must be a vector.
      * `x` and `y` must have the same number of dimensions.
      * The outermost dimensions of `condition`, `x`, and `y` must all have the
        same size.
      * `result[i] = x[i]` if `condition[i]` is true.
      * `result[i] = y[i]` if `condition[i]` is false.

  Args:
    condition: A potentially ragged tensor of type `bool`
    x: A potentially ragged tensor (optional).
    y: A potentially ragged tensor (optional).  Must be specified if `x` is
      specified.  Must have the same rank and type as `x`.
    name: A name of the operation (optional)

  Returns:
    : If both `x` and `y` are `None`:
      A `Tensor` with shape `(num_true, dim_size(condition))`.
    : Otherwise:
      A potentially ragged tensor with the same type, rank, and outermost
      dimension size as `x` and `y`.
      `result.ragged_rank = max(x.ragged_rank, y.ragged_rank)`.

  Raises:
    ValueError: When exactly one of `x` or `y` is non-`None`; or when
      `condition`, `x`, and `y` have incompatible shapes.

  #### Examples:

  >>> # Coordinates where condition is true.
  >>> condition = tf.ragged.constant([[True, False, True], [False, True]])
  >>> print(where(condition))
  tf.Tensor( [[0 0] [0 2] [1 1]], shape=(3, 2), dtype=int64)

  >>> # Elementwise selection between x and y, based on condition.
  >>> condition = tf.ragged.constant([[True, False, True], [False, True]])
  >>> x = tf.ragged.constant([['A', 'B', 'C'], ['D', 'E']])
  >>> y = tf.ragged.constant([['a', 'b', 'c'], ['d', 'e']])
  >>> print(where(condition, x, y))
  <tf.RaggedTensor [[b'A', b'b', b'C'], [b'd', b'E']]>

  >>> # Row selection between x and y, based on condition.
  >>> condition = [True, False]
  >>> x = tf.ragged.constant([['A', 'B', 'C'], ['D', 'E']])
  >>> y = tf.ragged.constant([['a', 'b', 'c'], ['d', 'e']])
  >>> print(where(condition, x, y))
  <tf.RaggedTensor [[b'A', b'B', b'C'], [b'd', b'e']]>
  """
  ...

