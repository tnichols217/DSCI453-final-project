"""
This type stub file was generated by pyright.
"""

from typing import List, Optional
from tensorflow.python.framework import tensor_spec

"""Implmentation for defining get_compiler_ir."""
def maybe_get_device_name(device_name):
  ...

def make_handledata_tensor_specs(resource_vars): # -> list[TensorSpec]:
  """Convert tf.Variable list to its corresponding TensorSpec list."""
  ...

def from_concrete_function(concrete_fn, specialized_flat_specs: Optional[List[tensor_spec.TensorSpec]] = ...): # -> Callable[..., Any]:
  """Generate the Compiler Ir from tf concrete function with TensorSpec.

  Args:
    concrete_fn: returned by using get_concrete_function.
    specialized_flat_specs: specialized flat tf.TensorSpecs for function args.

  Returns:
    Function callable that generate the HLO text.

  Raises:
      ValueError: if concrete_fn is not "compilable" without concrete
      inputs.
  """
  ...

