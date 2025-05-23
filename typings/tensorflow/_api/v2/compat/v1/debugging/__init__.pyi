"""
This type stub file was generated by pyright.
"""

import sys as _sys
from tensorflow._api.v2.compat.v1.debugging import experimental
from tensorflow.python.ops.gen_array_ops import check_numerics
from tensorflow.python.ops.gen_math_ops import is_finite, is_inf, is_nan
from tensorflow.python.debug.lib.check_numerics_callback import disable_check_numerics, enable_check_numerics
from tensorflow.python.eager.context import get_log_device_placement, set_log_device_placement
from tensorflow.python.ops.check_ops import assert_equal, assert_greater, assert_greater_equal, assert_integer, assert_less, assert_less_equal, assert_near, assert_negative, assert_non_negative, assert_non_positive, assert_none_equal, assert_positive, assert_proper_iterable, assert_rank, assert_rank_at_least, assert_rank_in, assert_same_float_dtype, assert_scalar, assert_shapes, assert_type, is_non_decreasing, is_numeric_tensor, is_strictly_increasing
from tensorflow.python.ops.control_flow_assert import Assert
from tensorflow.python.ops.numerics import verify_tensor_all_finite as assert_all_finite
from tensorflow.python.util.traceback_utils import disable_traceback_filtering, enable_traceback_filtering, is_traceback_filtering_enabled
from tensorflow.python.util import module_wrapper as _module_wrapper

"""Public API for tf._api.v2.debugging namespace
"""
if not isinstance(_sys.modules[__name__], _module_wrapper.TFModuleWrapper):
  ...
