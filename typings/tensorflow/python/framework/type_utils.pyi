"""
This type stub file was generated by pyright.
"""

"""Utility functions for types information, incuding full type information."""
_DT_TO_FT = ...
def fulltypes_for_flat_tensors(element_spec): # -> list[None] | object:
  """Convert the element_spec for a dataset to a list of FullType Def.

  Note that "flat" in this function and in `_flat_tensor_specs` is a nickname
  for the "batchable tensor list" encoding used by datasets and map_fn.
  The FullTypeDef created corresponds to this encoding (e.g. that uses variants
  and not the FullTypeDef corresponding to the default "component" encoding).

  This is intended for temporary internal use and expected to be removed
  when type inference support is sufficient. See limitations of
  `_translate_to_fulltype_for_flat_tensors`.

  Args:
    element_spec: A nest of TypeSpec describing the elements of a dataset (or
      map_fn).

  Returns:
    A list of FullTypeDef correspoinding to ELEMENT_SPEC. The items
    in this list correspond to the items in `_flat_tensor_specs`.
  """
  ...

def fulltype_list_to_product(fulltype_list):
  """Convert a list of FullType Def into a single FullType Def."""
  ...

def iterator_full_type_from_spec(element_spec):
  """Returns a FullTypeDef for an iterator for the elements.

  Args:
     element_spec: A nested structure of `tf.TypeSpec` objects representing the
       element type specification.

  Returns:
    A FullTypeDef for an iterator for the element tensor representation.
  """
  ...

