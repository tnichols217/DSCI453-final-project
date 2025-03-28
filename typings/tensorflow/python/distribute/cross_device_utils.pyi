"""
This type stub file was generated by pyright.
"""

from typing import List, Optional, Union
from tensorflow.python.distribute import collective_util
from tensorflow.python.framework import indexed_slices, ops
from tensorflow.python.types import core

"""Utilities for cross_device_ops."""
INSTANCE_KEY_START_NUMBER = ...
def aggregate_gradients_using_nccl(replica_grads): # -> list[Any]:
  """Aggregate gradients using nccl allreduce."""
  ...

def aggregate_gradients_using_hierarchical_copy(avail_devices, replica_grads): # -> list[Any]:
  """Aggregate gradients using hierarchical copies.

  Args:
    avail_devices: available GPU devices.
    replica_grads: List of lists of (gradient, variable) tuples. The outer list
      is over replicas. The inner list is over individual gradients.

  Returns:
    The list of (aggregated_gradient, variable), where the gradient has been
      summed across all replicas and the variable is chosen from the first
      replica.
  """
  ...

def aggregate_single_gradient_using_copy(grad_and_vars, use_mean, check_inf_nan): # -> tuple[tuple[Any | defaultdict[Any, Any] | list[Any] | object | SymbolicTensor | None, Any], Any] | tuple[tuple[Any | defaultdict[Any, Any] | list[Any] | object | SymbolicTensor | None, Any], None]:
  """Calculate the average gradient for a shared variable across all replicas.

  Note that this function provides a synchronization point across all replicas.

  Args:
    grad_and_vars: A list or tuple of (gradient, variable) tuples. Each
      (gradient, variable) pair within the outer list represents the gradient
      of the variable calculated for a single replica, and the number of pairs
      equals the number of replicas.
    use_mean: if True, mean is taken, else sum of gradients is taken.
    check_inf_nan: check grads for nans and infs.

  Returns:
    The tuple ([(average_gradient, variable),], has_nan_or_inf) where the
      gradient has been averaged across all replicas. The variable is chosen
      from the first replica. The has_nan_or_inf indicates the grads has nan or
      inf.
  """
  ...

class CollectiveKeys:
  """Class that manages collective keys.

  We need to manage three different keys for collective:

  *Group key*: an integer key to identify the set of cooperative devices.
  Collective ops work under the same set of devices must using the same group
  key.

  *Instance key*: an integer key to identify the set of same counterpart of
  tensors on different devices in a device group that need to be all-reduced.

  This class is thread safe.
  """
  def __init__(self, group_key_start=...) -> None:
    """Initializes the object.

    Args:
      group_key_start: the starting integer of group key.
    """
    ...
  
  def get_group_key(self, devices):
    """Returns a group key for the list of local devices.

    The same group key is returned if the list of local devices is the same.

    Args:
      devices: a list of local canonical device strings in a collective group.

    Returns:
      a group key.
    """
    ...
  
  def get_instance_key(self, group_key, device):
    """Returns a new instance key for use in defining a collective op.

    You should call this once per each collective op of a collective instance.

    Args:
      group_key: the group key returned by get_group_key(). You should not
        assign the group key yourself.
      device: a canonical device string. It should be the device this collective
        op is on.

    Returns:
      a new instance key.

    Raises:
      ValueError: when the group key is invalid or the device is not in the
      group.
    """
    ...
  
  def __deepcopy__(self, memo): # -> CollectiveKeys:
    ...
  


class CollectiveReplicaLauncher:
  """Launch collectives on one replica."""
  _prefer_unique_instance_key = ...
  _prefer_ordering_token = ...
  def __init__(self, group_key: int, group_size: int, collective_keys: CollectiveKeys, device: str, options: collective_util.Options) -> None:
    ...
  
  def can_order_nccl(self): # -> bool:
    """Whether this launcher can order NCCL operations."""
    ...
  
  def all_reduce(self, input_tensor: core.TensorLike, control_input: Optional[Union[core.TensorLike, ops.Operation]] = ..., options: Optional[collective_util.Options] = ...) -> core.Tensor:
    """All-reduce a dense tensor.

    Args:
      input_tensor: a dense tensor. It must have the same shape on all replicas.
      control_input: if not None, add control edges between control_input and
        the all-reduce.
      options: an optional tf.distribute.experimental.CommunicationOptions. If
        provided, it overrides the default options.

    Returns:
      The reduced tensor.
    """
    ...
  
  def batch_all_reduce(self, input_tensor_packs: List[List[core.TensorLike]], options: Optional[collective_util.Options] = ...) -> core.Tensor:
    """Batch all-reduce dense tensors.

    This takes a list of batches of tensors. Using multiple batches have the
    benefit that it doesn't need to wait for all inputs to be ready to start the
    all-reduce.

    Args:
      input_tensor_packs: a list of lists of dense tensors.
      options: an optional tf.distribute.experimental.CommunicationOptions. If
        provided, it overrides the default options.

    Returns:
      A flat list of reduced tensors.
    """
    ...
  
  def all_gather(self, input_tensor: core.TensorLike, axis: core.TensorLike, options: Optional[collective_util.Options] = ...) -> core.Tensor:
    """All-gather a dense tensor.

    This method must be called inside a tf.function.

    Args:
      input_tensor: a dense tensor. It must have the same rank on all replicas,
        and dimensions other than `axis` need to be the same as well.
      axis: 0-D int32 Tensor. Dimension along which to gather. Must be in the
        range [0, rank(value)).
      options: an optional tf.distribute.experimental.CommunicationOptions. If
        provided, it overrides the default options.

    Returns:
      The gathered Tensor.

    Raises:
      RuntimeError: if called in eager mode.
    """
    ...
  
  def all_reduce_indexed_slices(self, input_slices: indexed_slices.IndexedSlices, options: Optional[collective_util.Options] = ...) -> indexed_slices.IndexedSlices:
    """All-reduce an IndexedSlices.

    This method can be called outside  tf.function.

    Args:
      input_slices: an IndexedSlices.
      options: an optional tf.distribute.experimental.CommunicationOptions. If
        provided, it overrides the default options.

    Returns:
      The reduced IndexedSlices.
    """
    ...
  


def aggregate_tensors_or_indexed_slices(values, accumulation_fn=...): # -> defaultdict[Any, Any] | Any | list[Any] | object | SymbolicTensor | IndexedSlices | None:
  """Aggregate tensors using `accumulation_fn` and IndexedSlices via concat."""
  ...

def divide_by_n_tensors_or_indexed_slices(value, n): # -> IndexedSlices:
  ...

def copy_tensor_or_indexed_slices_to_device(value, device): # -> IndexedSlices | defaultdict[Any, Any] | Any | list[Any] | object | None:
  """Copies a tensor or IndexedSlices to a device."""
  ...

def is_indexed_slices(value): # -> bool:
  ...

def split_by_sparsity(values): # -> tuple[list[Any], list[Any], list[Any], list[Any]]:
  """Split values into dense and sparse values.

  Args:
    values: a list of tensors or `PerReplica`s.

  Returns:
    Four lists:
      a list of dense values, a list of their indices in `values` and
      a list of sparse values, a list of their indices in `values`.
  """
  ...

def stitch_values(values_and_indices_list): # -> list[None]:
  """Stitch values together according to their indices.

  Args:
    values_and_indices_list: a list of tuples of values and indices indicating
      the values and positions in the returned list.

  Returns:
    a stitched list of values.
  """
  ...

def group_by_size(input_tensors, bytes_per_pack): # -> list[Any]:
  """Groups `input_tensors` into chunks of `bytes_per_pack`.

  The method preserves the original order of `input_tensors`. The grouping is
  best effort, each pack could have more or less bytes than `bytes_per_pack`.
  It only groups values with known shape.

  Args:
    input_tensors: a list of Tensor.
    bytes_per_pack: an integer.

  Returns:
    A list of packs of Tensor. All values are grouped into one pack if
    `bytes_per_pack` is zero or any of the value has unknown shape.
  """
  ...

