"""
This type stub file was generated by pyright.
"""

"""Utility functions used by values.py and ps_values.py."""
def write_object_proto(var, proto, options): # -> None:
  """Update a SavedObject proto for the caller.

  If a DistributedVariable object supports this method, it will be called when
  saving with a pre-built `SavedObject` proto representing the object, plus an
  instance of `SaveOptions`. This method is then free to modify that proto
  instance.

  `DistributedVariable` with `AUTO` or `ON_WRITE` synchronization optionally
   write out information about their components to the
   `experimental_distributed_variable_components` field of a
   `SavedVariable` (depending on the `SaveOptions` variable policy).

  Args:
    var: The DistributedVariable object.
    proto: A pre-built `SavedObject` proto for this object. It is assumed this
      will be a `SavedVariable` instance.
    options: A `SaveOptions` instance.
  """
  ...

def get_on_write_saveable(var, primary_var, name): # -> tuple[Callable[[], Any | None], list[SaveSpec]]:
  """Return saveable spec for AUTO and ON_WRITE variables."""
  ...

def get_on_write_restore_ops(var, tensor): # -> object | _dispatcher_for_no_op | Operation | None:
  """Return restore ops for AUTO and ON_WRITE variables."""
  ...

def get_on_read_saveable(var, primary_var, name): # -> tuple[Callable[[], Any], list[SaveSpec]]:
  """Return saveables for ON_READ variable."""
  ...

def get_on_read_restore_ops(var, tensor, aggregation): # -> object | _dispatcher_for_no_op | Operation | None:
  """Return restore ops for ON_READ variables."""
  ...

def in_replica_update_context(): # -> bool:
  ...

def on_write_assign(var, value, use_locking=..., name=..., read_value=...):
  ...

def on_write_assign_add(var, value, use_locking=..., name=..., read_value=...):
  ...

def on_write_assign_sub(var, value, use_locking=..., name=..., read_value=...):
  ...

def assign_on_each_device(var, assign_func, value, read_value): # -> object | _dispatcher_for_no_op | Operation | None:
  """Update the variable on each replica with the given assign_func and value."""
  ...

def on_read_assign_sub_cross_replica(var, value, read_value=...): # -> object | _dispatcher_for_no_op | Operation | None:
  ...

def on_read_assign_add_cross_replica(var, value, read_value=...): # -> object | _dispatcher_for_no_op | Operation | None:
  ...

def on_read_assign_cross_replica(var, value, read_value=...): # -> object | _dispatcher_for_no_op | Operation | None:
  """Return the value of the variable in cross replica context."""
  ...

def scatter_sub(var, sparse_delta, use_locking=..., name=...):
  ...

def scatter_add(var, sparse_delta, use_locking=..., name=...):
  ...

def scatter_mul(var, sparse_delta, use_locking=..., name=...):
  ...

def scatter_div(var, sparse_delta, use_locking=..., name=...):
  ...

def scatter_min(var, sparse_delta, use_locking=..., name=...):
  ...

def scatter_max(var, sparse_delta, use_locking=..., name=...):
  ...

def scatter_update(var, sparse_delta, use_locking=..., name=...):
  ...

def get_current_replica_id_as_int(): # -> int | Any | None:
  """Returns the current replica ID as an integer, or `None`."""
  ...

def assign_on_device(device, variable, tensor):
  ...

def assign_add_on_device(device, variable, tensor):
  ...

def assign_sub_on_device(device, variable, tensor):
  ...

def assert_replica_context(strategy): # -> None:
  ...

def apply_aggregation(strategy, value, aggregation, destinations):
  ...

aggregation_error_msg = ...
scatter_error_msg = ...
def is_saving_non_distributed(): # -> Literal[False]:
  """Returns whether we're saving a non-distributed version of the model.

  It returns True iff we are in saving context and are saving a non-distributed
  version of the model. That is, SaveOptions.experimental_variable_policy is
  NONE.

  Returns:
    A boolean.
  """
  ...

def mark_as_unsaveable(): # -> None:
  """Marks the function as unsaveable if not inside save context."""
  ...

