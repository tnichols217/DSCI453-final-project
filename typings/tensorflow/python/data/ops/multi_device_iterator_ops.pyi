"""
This type stub file was generated by pyright.
"""

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import composite_tensor, type_spec

"""Python wrapper for prefetching_ops."""
class _PerDeviceGenerator(dataset_ops.DatasetV2):
  """A `dummy` generator dataset."""
  def __init__(self, shard_num, multi_device_iterator_resource, incarnation_id, source_device, element_spec, iterator_is_anonymous) -> None:
    ...
  
  @property
  def element_spec(self): # -> Any:
    ...
  


class _ReincarnatedPerDeviceGenerator(dataset_ops.DatasetV2):
  """Creates a _PerDeviceGenerator-like dataset with a new incarnation_id.

  Re-uses the functions from the provided per_device_dataset and just switches
  out the function argument corresponding to the incarnation_id.
  """
  def __init__(self, per_device_dataset, incarnation_id) -> None:
    ...
  
  @property
  def element_spec(self):
    ...
  


class MultiDeviceIterator:
  """An iterator over multiple devices."""
  def __init__(self, dataset, devices, max_buffer_size=..., prefetch_buffer_size=..., source_device=...) -> None:
    """Constructs a MultiDeviceIterator.

    Args:
      dataset: The input dataset to be iterated over.
      devices: The list of devices to fetch data to.
      max_buffer_size: Maximum size of the host side per device buffer to keep.
      prefetch_buffer_size: if > 0, then we setup a buffer on each device to
        prefetch into.
      source_device: The host device to place the `dataset` on.  In order to
        prevent deadlocks, if the prefetch_buffer_size is greater than the
        max_buffer_size, we set the max_buffer_size to prefetch_buffer_size.
    """
    ...
  
  def get_next(self, device=...): # -> list[Any]:
    """Returns the next element given a `device`, else returns all in a list."""
    ...
  
  def get_next_as_optional(self): # -> list[Any]:
    ...
  
  @property
  def initializer(self): # -> object | _dispatcher_for_no_op | Operation | None:
    ...
  
  @property
  def element_spec(self):
    ...
  


class MultiDeviceIteratorSpec(type_spec.TypeSpec):
  """Type specification for `OwnedMultiDeviceIterator`."""
  __slots__ = ...
  def __init__(self, devices, source_device, element_spec) -> None:
    ...
  
  @property
  def value_type(self): # -> type[OwnedMultiDeviceIterator]:
    ...
  
  @staticmethod
  def from_value(value): # -> MultiDeviceIteratorSpec:
    ...
  


class OwnedMultiDeviceIterator(composite_tensor.CompositeTensor):
  """An iterator over multiple devices.

  The multi-device iterator resource created through `OwnedMultiDeviceIterator`
  is owned by the Python object and the life time of the underlying resource is
  tied to the life time of the `OwnedMultiDeviceIterator` object. This makes
  `OwnedMultiDeviceIterator` appropriate for use in eager mode and inside of
  tf.functions.
  """
  def __init__(self, dataset=..., devices=..., max_buffer_size=..., prefetch_buffer_size=..., source_device=..., components=..., element_spec=...) -> None:
    """Constructs an owned MultiDeviceIterator object.

    Args:
      dataset: The input dataset to be iterated over.
      devices: (Required.) The list of devices to fetch data to.
      max_buffer_size: Maximum size of the host side per device buffer to keep.
      prefetch_buffer_size: if > 0, then we setup a buffer on each device to
        prefetch into.
      source_device: The host device to place the `dataset` on.  In order to
        prevent deadlocks, if the prefetch_buffer_size is greater than the
        max_buffer_size, we set the max_buffer_size to prefetch_buffer_size.
      components: Tensor components to construct the MultiDeviceIterator from.
      element_spec: A (nested) structure of `tf.TypeSpec` objects that
        represents the type specification of elements of the iterator.

    Raises:
      RuntimeError: If executed in graph mode or outside of function building
        mode.
      ValueError: If any of the following happens:
        - `devices` is `None`
        - `dataset` is `None` and either `components` or `element_spec` is
          `None`
        - `dataset` is not None and either `components` or `element_spec` is
          provided
    """
    ...
  
  def get_next(self, device=...): # -> list[Any]:
    """Returns the next element given a `device`, else returns all in a list."""
    ...
  
  def __iter__(self): # -> Self:
    ...
  
  def next(self): # -> list[Any]:
    ...
  
  def __next__(self): # -> list[Any]:
    ...
  
  def get_next_as_optional(self): # -> list[Any]:
    ...
  
  @property
  def element_spec(self):
    ...
  


