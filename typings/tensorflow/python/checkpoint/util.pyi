"""
This type stub file was generated by pyright.
"""

"""Utilities for extracting and writing checkpoint info`."""
def serialize_slot_variables(trackable_objects, node_ids, object_names): # -> ObjectIdentityDictionary:
  """Gather and name slot variables."""
  ...

def get_mapped_trackable(trackable, object_map):
  """Returns the mapped trackable if possible, otherwise returns trackable."""
  ...

def get_full_name(var): # -> Literal['']:
  """Gets the full name of variable for name-based checkpoint compatibility."""
  ...

def add_checkpoint_values_check(object_graph_proto): # -> None:
  """Determines which objects have checkpoint values and save this to the proto.

  Args:
    object_graph_proto: A `TrackableObjectGraph` proto.
  """
  ...

def objects_ids_and_slot_variables_and_paths(graph_view, skip_slot_variables=...): # -> tuple[Any, Any, ObjectIdentityDictionary, ObjectIdentityDictionary, ObjectIdentityDictionary]:
  """Traverse the object graph and list all accessible objects.

  Looks for `Trackable` objects which are dependencies of
  `root_trackable`. Includes slot variables only if the variable they are
  slotting for and the optimizer are dependencies of `root_trackable`
  (i.e. if they would be saved with a checkpoint).

  Args:
    graph_view: A GraphView object.
    skip_slot_variables: If True does not return trackables for slot variable.
      Default False.

  Returns:
    A tuple of (trackable objects, paths from root for each object,
                object -> node id, slot variables, object_names)
  """
  ...

def list_objects(graph_view, skip_slot_variables=...):
  """Traverse the object graph and list all accessible objects."""
  ...

