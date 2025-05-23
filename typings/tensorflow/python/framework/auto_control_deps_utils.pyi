"""
This type stub file was generated by pyright.
"""

"""Utilities for AutomaticControlDependencies."""
READ_ONLY_RESOURCE_INPUTS_ATTR = ...
RESOURCE_READ_OPS = ...
COLLECTIVE_MANAGER_IDS = ...
def register_read_only_resource_op(op_type): # -> None:
  """Declares that `op_type` does not update its touched resource."""
  ...

def get_read_only_resource_input_indices_graph(func_graph): # -> list[Any]:
  """Returns sorted list of read-only resource indices in func_graph.inputs."""
  ...

def get_read_write_resource_inputs(op): # -> tuple[ObjectIdentitySet, ObjectIdentitySet]:
  """Returns a tuple of resource reads, writes in op.inputs.

  Args:
    op: Operation

  Returns:
    A 2-tuple of ObjectIdentitySets, the first entry containing read-only
    resource handles and the second containing read-write resource handles in
    `op.inputs`.
  """
  ...

