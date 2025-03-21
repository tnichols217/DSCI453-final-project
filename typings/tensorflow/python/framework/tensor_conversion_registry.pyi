"""
This type stub file was generated by pyright.
"""

from tensorflow.python.util.tf_export import tf_export

"""Registry for tensor conversion functions."""
_tensor_conversion_func_registry = ...
_tensor_conversion_func_cache = ...
_tensor_conversion_func_lock = ...
_CONSTANT_OP_CONVERTIBLES = ...
def register_tensor_conversion_function_internal(base_type, conversion_func, priority=...): # -> None:
  """Internal version of register_tensor_conversion_function.

  See docstring of `register_tensor_conversion_function` for details.

  The internal version of the function allows registering conversions
  for types in the _UNCONVERTIBLE_TYPES tuple.

  Args:
    base_type: The base type or tuple of base types for all objects that
      `conversion_func` accepts.
    conversion_func: A function that converts instances of `base_type` to
      `Tensor`.
    priority: Optional integer that indicates the priority for applying this
      conversion function. Conversion functions with smaller priority values run
      earlier than conversion functions with larger priority values. Defaults to
      100.

  Raises:
    TypeError: If the arguments do not have the appropriate type.
  """
  ...

@tf_export("register_tensor_conversion_function")
def register_tensor_conversion_function(base_type, conversion_func, priority=...): # -> None:
  """Registers a function for converting objects of `base_type` to `Tensor`.

  The conversion function must have the following signature:

  ```python
      def conversion_func(value, dtype=None, name=None, as_ref=False):
        # ...
  ```

  It must return a `Tensor` with the given `dtype` if specified. If the
  conversion function creates a new `Tensor`, it should use the given
  `name` if specified. All exceptions will be propagated to the caller.

  The conversion function may return `NotImplemented` for some
  inputs. In this case, the conversion process will continue to try
  subsequent conversion functions.

  If `as_ref` is true, the function must return a `Tensor` reference,
  such as a `Variable`.

  NOTE: The conversion functions will execute in order of priority,
  followed by order of registration. To ensure that a conversion function
  `F` runs before another conversion function `G`, ensure that `F` is
  registered with a smaller priority than `G`.

  Args:
    base_type: The base type or tuple of base types for all objects that
      `conversion_func` accepts.
    conversion_func: A function that converts instances of `base_type` to
      `Tensor`.
    priority: Optional integer that indicates the priority for applying this
      conversion function. Conversion functions with smaller priority values run
      earlier than conversion functions with larger priority values. Defaults to
      100.

  Raises:
    TypeError: If the arguments do not have the appropriate type.
  """
  ...

def get(query): # -> list[Any]:
  """Get conversion function for objects of `cls`.

  Args:
    query: The type to query for.

  Returns:
    A list of conversion functions in increasing order of priority.
  """
  ...

def convert(value, dtype=..., name=..., as_ref=..., preferred_dtype=..., accepted_result_types=...): # -> Any | Tensor:
  """Converts `value` to a `Tensor` using registered conversion functions.

  Args:
    value: An object whose type has a registered `Tensor` conversion function.
    dtype: Optional element type for the returned tensor. If missing, the type
      is inferred from the type of `value`.
    name: Optional name to use if a new `Tensor` is created.
    as_ref: Optional boolean specifying if the returned value should be a
      reference-type `Tensor` (e.g. Variable). Pass-through to the registered
      conversion function. Defaults to `False`.
    preferred_dtype: Optional element type for the returned tensor.
      Used when dtype is None. In some cases, a caller may not have a dtype
      in mind when converting to a tensor, so `preferred_dtype` can be used
      as a soft preference. If the conversion to `preferred_dtype` is not
      possible, this argument has no effect.
    accepted_result_types: Optional collection of types as an allow-list
      for the returned value. If a conversion function returns an object
      which is not an instance of some type in this collection, that value
      will not be returned.

  Returns:
    A `Tensor` converted from `value`.

  Raises:
    ValueError: If `value` is a `Tensor` and conversion is requested
      to a `Tensor` with an incompatible `dtype`.
    TypeError: If no conversion function is registered for an element in
      `values`.
    RuntimeError: If a registered conversion function returns an invalid
      value.
  """
  ...

