"""
This type stub file was generated by pyright.
"""

from tensorflow.python.eager import def_function

"""Tools for deserializing `Function`s."""
def set_preinitialized_function_spec(concrete_fn, spec): # -> None:
  """Set the FunctionType of the ConcreteFunction using FunctionSpec."""
  ...

def setup_bare_concrete_function(saved_bare_concrete_function, concrete_functions):
  """Makes a restored bare concrete function callable."""
  ...

class RestoredFunction(def_function.Function):
  """Wrapper class for a function that has been restored from saved state.

  See `def_function.Function`.
  """
  def __init__(self, python_function, name, function_spec, concrete_functions) -> None:
    ...
  


def recreate_function(saved_function, concrete_functions):
  """Creates a `Function` from a `SavedFunction`.

  Args:
    saved_function: `SavedFunction` proto.
    concrete_functions: map from function name to `ConcreteFunction`. As a side
      effect of this function, the `FunctionSpec` from `saved_function` is added
      to each `ConcreteFunction` in this map.

  Returns:
    A `Function`.
  """
  ...

def load_function_def_library(library, saved_object_graph=..., load_shared_name_suffix=..., wrapper_function=...): # -> dict[Any, Any]:
  """Load a set of functions as concrete functions without captured inputs.

  Functions names are manipulated during load such that they do not overlap
  with previously created ones.

  Gradients are re-registered under new names. Ops that reference the gradients
  are updated to reflect the new registered names.

  Args:
    library: FunctionDefLibrary proto message.
    saved_object_graph: SavedObjectGraph proto message. If not passed in,
      concrete function structured signatures and outputs will not be set.
    load_shared_name_suffix: If specified, used to uniquify shared names.
      Otherwise, a unique name is generated.
    wrapper_function: An object that will be wrapped on newly created functions.

  Returns:
    Map of original function names in the library to instances of
    `ConcreteFunction` without captured inputs.

  Raises:
    ValueError: if functions dependencies have a cycle.
  """
  ...

def fix_node_def(node_def, functions, shared_name_suffix): # -> None:
  """Replace functions calls and shared names in `node_def`."""
  ...

_FUNCTION_WRAPPER_NAME_REGEX = ...
