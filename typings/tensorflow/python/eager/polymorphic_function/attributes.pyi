"""
This type stub file was generated by pyright.
"""

"""This file lists FunctionDef attributes and corresponding allowlists."""
API_IMPLEMENTS = ...
API_PREFERRED_DEVICE = ...
BACKWARD_FUNCTION = ...
DISABLE_ACD = ...
DISABLE_CALL_SHAPE_INFERENCE = ...
DISABLE_SUMMARIES_AT_RUNTIME = ...
EAGER_RUNTIME_CONSTRUCTION_CONTEXT = ...
FORWARD_FUNCTION = ...
GO_BACKWARDS = ...
IMPLEMENTS = ...
INPUT_SHAPES = ...
INTS_ON_DEVICE = ...
NO_INLINE = ...
ORIGINAL_FUNCTION_NAME = ...
OUTPUTS_ON_OP_DEVICE = ...
QUANTIZED_COMPOSITE_FUNCTION = ...
QUANTIZED_OPS = ...
RUNTIME_CONSTANT_OPTIMIZATION = ...
SHARED_RENDEZVOUS = ...
TF_DATA_FUNCTION = ...
TFTRT_ALLOW_BUILD_AT_RUNTIME = ...
TFTRT_CONVERT_FUNCTION = ...
TFTRT_IS_DYN_OP = ...
TFTRT_LOGGER = ...
TFTRT_MAX_BATCH_SIZE = ...
TFTRT_MAX_CACHED_ENGINES = ...
TFTRT_MAX_WORKSPACE_SIZE = ...
TFTRT_MIN_SEGMENT_SIZE = ...
TFTRT_PRECISION_MODE = ...
TFTRT_PROFILE_STRATEGY = ...
TFTRT_USE_CALIBRATION = ...
TFTRT_USE_IMPLICIT_BATCH = ...
TIME_MAJOR = ...
XLA_COMPILE = ...
XLA_COMPILE_OPTIONAL = ...
XLA_SCOPE = ...
XLA_SEPERATE_COMPILED_GRADIENTS = ...
POLYMORPHIC_FUNCTION_ALLOWLIST = ...
TRACING_COMPILATION_ALLOWLIST = ...
MONOMORPHIC_FUNCTION_ALLOWLIST = ...
def parse_func_attrs(attributes, allowlist=...): # -> dict[Any, Any]:
  """Convert the keyword arguments into function_def attributes.

  Currently only support primitive types: bool, int, float and string.

  Args:
    attributes: the dictionary of attributes.
    allowlist: set of attribute names allowed.
  Returns:
    A dict of attributes where the key is the name of attribute and the value
      is the AttrValue proto.
  Raises:
    ValueError: If the kwargs contains unallowlisted name or unsupported value
      types.
  """
  ...

