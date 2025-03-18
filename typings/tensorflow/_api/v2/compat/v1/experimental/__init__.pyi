"""
This type stub file was generated by pyright.
"""

import sys as _sys
from tensorflow._api.v2.compat.v1.experimental import extension_type
from tensorflow.python.data.ops.optional_ops import Optional
from tensorflow.python.eager.context import async_clear_error, async_scope, function_executor_type
from tensorflow.python.framework.dtypes import float8_e4m3fn, float8_e5m2, int4, uint4
from tensorflow.python.framework.extension_type import BatchableExtensionType, ExtensionType, ExtensionTypeBatchEncoder, ExtensionTypeSpec
from tensorflow.python.framework.load_library import register_filesystem_plugin
from tensorflow.python.framework.strict_mode import enable_strict_mode
from tensorflow.python.ops.control_flow_util_v2 import set_output_all_intermediates as output_all_intermediates
from tensorflow.python.ops.ragged.dynamic_ragged_shape import DynamicRaggedShape
from tensorflow.python.ops.ragged.row_partition import RowPartition
from tensorflow.python.ops.structured.structured_tensor import StructuredTensor
from tensorflow.python.util.dispatch import dispatch_for_api, dispatch_for_binary_elementwise_apis, dispatch_for_binary_elementwise_assert_apis, dispatch_for_unary_elementwise_apis, unregister_dispatch_for
from tensorflow.python.util import module_wrapper as _module_wrapper

"""Public API for tf._api.v2.experimental namespace
"""
if not isinstance(_sys.modules[__name__], _module_wrapper.TFModuleWrapper):
  ...
