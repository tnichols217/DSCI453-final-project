"""
This type stub file was generated by pyright.
"""

from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.tf_export import tf_export

"""Utilities for manipulating the loss collections."""
def squeeze_or_expand_dimensions(y_pred, y_true=..., sample_weight=...): # -> tuple[Any | SymbolicTensor | defaultdict[Any, Any] | list[Any] | tuple[Any, ...] | None, Any | SymbolicTensor | defaultdict[Any, Any] | list[Any] | tuple[Any, ...] | None] | tuple[Any | SymbolicTensor | defaultdict[Any, Any] | list[Any] | tuple[Any, ...] | None, Any | SymbolicTensor | defaultdict[Any, Any] | list[Any] | tuple[Any, ...] | None, Any] | tuple[Any | SymbolicTensor | defaultdict[Any, Any] | list[Any] | tuple[Any, ...] | None, Any | SymbolicTensor | defaultdict[Any, Any] | list[Any] | tuple[Any, ...] | None, Any | defaultdict[Any, Any] | list[Any] | tuple[Any, ...] | None]:
  """Squeeze or expand last dimension if needed.

  1. Squeezes last dim of `y_pred` or `y_true` if their rank differs by 1
  (using `confusion_matrix.remove_squeezable_dimensions`).
  2. Squeezes or expands last dim of `sample_weight` if its rank differs by 1
  from the new rank of `y_pred`.
  If `sample_weight` is scalar, it is kept scalar.

  This will use static shape if available. Otherwise, it will add graph
  operations, which could result in a performance hit.

  Args:
    y_pred: Predicted values, a `Tensor` of arbitrary dimensions.
    y_true: Optional label `Tensor` whose dimensions match `y_pred`.
    sample_weight: Optional weight scalar or `Tensor` whose dimensions match
      `y_pred`.

  Returns:
    Tuple of `y_pred`, `y_true` and `sample_weight`. Each of them possibly has
    the last dimension squeezed,
    `sample_weight` could be extended by one dimension.
    If `sample_weight` is None, (y_pred, y_true) is returned.
  """
  ...

def scale_losses_by_sample_weight(losses, sample_weight):
  """Scales loss values by the given sample weights.

  `sample_weight` dimensions are updated to match with the dimension of `losses`
  if possible by using squeeze/expand/broadcast.

  Args:
    losses: Loss tensor.
    sample_weight: Sample weights tensor.

  Returns:
    `losses` scaled by `sample_weight` with dtype float32.
  """
  ...

@tf_contextlib.contextmanager
def check_per_example_loss_rank(per_example_loss): # -> Generator[None, Any, None]:
  """Context manager that checks that the rank of per_example_loss is at least 1.

  Args:
    per_example_loss: Per example loss tensor.

  Yields:
    A context manager.
  """
  ...

@tf_export(v1=["losses.add_loss"])
def add_loss(loss, loss_collection=...): # -> None:
  """Adds a externally defined loss to the collection of losses.

  Args:
    loss: A loss `Tensor`.
    loss_collection: Optional collection to add the loss to.
  """
  ...

@tf_export(v1=["losses.get_losses"])
def get_losses(scope=..., loss_collection=...): # -> list[Any]:
  """Gets the list of losses from the loss_collection.

  Args:
    scope: An optional scope name for filtering the losses to return.
    loss_collection: Optional losses collection.

  Returns:
    a list of loss tensors.
  """
  ...

@tf_export(v1=["losses.get_regularization_losses"])
def get_regularization_losses(scope=...): # -> list[Any]:
  """Gets the list of regularization losses.

  Args:
    scope: An optional scope name for filtering the losses to return.

  Returns:
    A list of regularization losses as Tensors.
  """
  ...

@tf_export(v1=["losses.get_regularization_loss"])
def get_regularization_loss(scope=..., name=...): # -> defaultdict[Any, Any] | Any | list[Any] | object | SymbolicTensor | Operation | _EagerTensorBase | None:
  """Gets the total regularization loss.

  Args:
    scope: An optional scope name for filtering the losses to return.
    name: The name of the returned tensor.

  Returns:
    A scalar regularization loss.
  """
  ...

@tf_export(v1=["losses.get_total_loss"])
def get_total_loss(add_regularization_losses=..., name=..., scope=...): # -> defaultdict[Any, Any] | Any | list[Any] | object | SymbolicTensor | None:
  """Returns a tensor whose value represents the total loss.

  In particular, this adds any losses you have added with `tf.add_loss()` to
  any regularization losses that have been added by regularization parameters
  on layers constructors e.g. `tf.layers`. Be very sure to use this if you
  are constructing a loss_op manually. Otherwise regularization arguments
  on `tf.layers` methods will not function.

  Args:
    add_regularization_losses: A boolean indicating whether or not to use the
      regularization losses in the sum.
    name: The name of the returned tensor.
    scope: An optional scope name for filtering the losses to return. Note that
      this filters the losses added with `tf.add_loss()` as well as the
      regularization losses to that scope.

  Returns:
    A `Tensor` whose value represents the total loss.

  Raises:
    ValueError: if `losses` is not iterable.
  """
  ...

