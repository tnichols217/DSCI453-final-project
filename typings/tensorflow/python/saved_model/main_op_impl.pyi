"""
This type stub file was generated by pyright.
"""

from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export

"""SavedModel main op implementation."""
_DEPRECATION_MSG = ...
@tf_export(v1=['saved_model.main_op.main_op'])
@deprecation.deprecated(None, _DEPRECATION_MSG)
def main_op(): # -> object | _dispatcher_for_no_op | Operation | None:
  """Returns a main op to init variables and tables.

  Returns the main op including the group of ops that initializes all
  variables, initializes local variables and initialize all tables.

  Returns:
    The set of ops to be run as part of the main op upon the load operation.
  """
  ...

@tf_export(v1=['saved_model.main_op_with_restore', 'saved_model.main_op.main_op_with_restore'])
@deprecation.deprecated(None, _DEPRECATION_MSG)
def main_op_with_restore(restore_op_name): # -> object | _dispatcher_for_no_op | Operation | None:
  """Returns a main op to init variables, tables and restore the graph.

  Returns the main op including the group of ops that initializes all
  variables, initialize local variables, initialize all tables and the restore
  op name.

  Args:
    restore_op_name: Name of the op to use to restore the graph.

  Returns:
    The set of ops to be run as part of the main op upon the load operation.
  """
  ...

