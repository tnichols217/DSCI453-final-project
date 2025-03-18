"""
This type stub file was generated by pyright.
"""

from typing import Any, Mapping, Optional
from tensorflow.python.checkpoint import checkpoint_adapter
from tensorflow.python.framework import tensor
from tensorflow.python.trackable import base

"""Logic for restoring checkpointed values for Trackables."""
class CheckpointPosition:
  """Indicates a position within a `_CheckpointRestoreCoordinator`."""
  __slots__ = ...
  def __init__(self, checkpoint, proto_id) -> None:
    """Specify an object within a checkpoint.

    Args:
      checkpoint: A _CheckpointRestoreCoordinator object.
      proto_id: The index of this object in TrackableObjectGraph.nodes.
    """
    ...
  
  def restore(self, trackable, reader=...): # -> None:
    """Restore this value into `trackable`."""
    ...
  
  def bind_object(self, trackable): # -> bool:
    """Set a checkpoint<->object correspondence.

    Args:
      trackable: The object to record a correspondence for.

    Returns:
      True if this is a new assignment, False if this object has already been
      mapped to a checkpointed `Object` proto.
    Raises:
      AssertionError: If another object is already bound to the `Object` proto.
    """
    ...
  
  def update_resharding_callback(self, callback: checkpoint_adapter.ReshardCallback): # -> None:
    """Add a resharding callback to the checkpoint.

    This will be applied to the checkpoint value before being supplied to the
    restore ops.

    Args:
     callback: Reshard callback for resharding this checkpoint position. Maybe
       None.
    """
    ...
  
  def has_non_trivial_reshard_callback(self) -> bool:
    """Determine whether this value has a non-trivial resharding callback."""
    ...
  
  def is_simple_variable(self) -> bool:
    """Determine whether this value is restorable with a Tensor initializer."""
    ...
  
  def value_tensors(self, shape_and_slices: Optional[str] = ...) -> Mapping[str, tensor.Tensor]:
    """Create value `Tensor`s for this object's attributes.

    Does not require that the Python object has been created. Used for
    restore-on-create when executing eagerly.

    Args:
      shape_and_slices: A dict mapping from object attribute names to a shape
        and slice string that will be passed to a RestoreV2 op. If the dict is
        None or if an object attribute is not in the dict, the full tensor will
        be restored.

    Returns:
      A dictionary mapping from object attribute names to `Tensor`s.
    """
    ...
  
  def gather_ops_or_named_saveables(self): # -> tuple[list[Any], dict[Any, Any], list[Any], dict[Any, Any]] | tuple[list[Any], dict[Any, Any], list[Any], defaultdict[Any, dict[Any, Any]]]:
    """Looks up or creates SaveableObjects which don't have cached ops.

    Returns:
      A tuple of (
          existing_restore_ops: list,
          named_saveables: dict,
          python_positions: list,
          registered_savers: dict)
    """
    ...
  
  def restore_ops(self, reader=...): # -> list[Any]:
    """Create or fetch restore ops for this object's attributes.

    Requires that the `Trackable` Python object has been bound to an object
    ID in the checkpoint.

    Args:
      reader: A `CheckpointReader`. If None, a new instance will be created.

    Returns:
      A list of operations when graph building, or an empty list when executing
      eagerly.
    """
    ...
  
  @property
  def checkpoint(self): # -> Any:
    ...
  
  @property
  def trackable(self):
    ...
  
  @property
  def object_proto(self):
    ...
  
  @property
  def proto_id(self): # -> Any:
    ...
  
  @property
  def restore_uid(self):
    ...
  
  def __repr__(self): # -> str:
    ...
  
  def value_shape(self): # -> None:
    """The shape of the VARIABLE_VALUE tensor.

    Returns:
      If found a TensorShape object, otherwise None.
    """
    ...
  
  def get_registered_saver_name(self): # -> None:
    """Returns the registered saver name defined in the Checkpoint."""
    ...
  
  def create_slot_variable_position(self, optimizer_object: Any, variable: base.Trackable, slot_variable_id: str, slot_name: str, reshard_callback: Optional[checkpoint_adapter.ReshardCallback] = ...): # -> tuple[CheckpointPosition, Any] | tuple[None, None]:
    """Generates CheckpointPosition for a slot variable.

    Args:
      optimizer_object: Optimizer that owns the slot variable.
      variable: Variable associated with the slot variable.
      slot_variable_id: ID of the slot variable.
      slot_name: Name of the slot variable.
      reshard_callback: A callback object for resharding value from checkpoint
        at restore.

    Returns:
      If there is a slot variable in the `optimizer_object` that has not been
      bound to the checkpoint, this function returns a tuple of (
        new `CheckpointPosition` for the slot variable,
        the slot variable itself).
    """
    ...
  
  def create_child_position(self, node_id): # -> CheckpointPosition:
    ...
  


def restore_nodes(save_path, nodes_to_restore): # -> None:
  """Restores nodes from a dict.

  Requires that the `Trackable` Python object has been bound to an object
  ID in the checkpoint.

  Args:
    save_path: a string represents path to the checkpoint.
    nodes_to_restore: a dict maps `node_id` to `trackable` to be restored.
  """
  ...

_DeferredSlotVariableRestoration = ...
