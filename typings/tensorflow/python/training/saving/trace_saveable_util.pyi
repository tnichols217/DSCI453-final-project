"""
This type stub file was generated by pyright.
"""

"""Utilities for tracing save and restore functions for SaveableObjects."""
def trace_save_restore_function_map(obj, factory_data_list): # -> dict[Any, Any]:
  """Traces all save and restore functions in the provided factory list.

  Args:
    obj: `Trackable` object.
    factory_data_list: List of `_CheckpointFactoryData`.

  Returns:
    Dict mapping atttribute names to tuples of concrete save/restore functions.
  """
  ...

