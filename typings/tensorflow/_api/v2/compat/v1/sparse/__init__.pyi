"""
This type stub file was generated by pyright.
"""

import sys as _sys
from tensorflow.python.framework.sparse_tensor import SparseTensor
from tensorflow.python.ops.array_ops import sparse_mask as mask, sparse_placeholder as placeholder
from tensorflow.python.ops.data_flow_ops import SparseConditionalAccumulator
from tensorflow.python.ops.math_ops import sparse_segment_mean as segment_mean, sparse_segment_sqrt_n as segment_sqrt_n, sparse_segment_sum as segment_sum
from tensorflow.python.ops.sparse_ops import from_dense, sparse_add as add, sparse_bincount as bincount, sparse_concat as concat, sparse_cross as cross, sparse_cross_hashed as cross_hashed, sparse_expand_dims as expand_dims, sparse_eye as eye, sparse_fill_empty_rows as fill_empty_rows, sparse_maximum as maximum, sparse_merge as merge, sparse_minimum as minimum, sparse_reduce_max as reduce_max, sparse_reduce_max_sparse as reduce_max_sparse, sparse_reduce_sum as reduce_sum, sparse_reduce_sum_sparse as reduce_sum_sparse, sparse_reorder as reorder, sparse_reset_shape as reset_shape, sparse_reshape as reshape, sparse_retain as retain, sparse_slice as slice, sparse_softmax as softmax, sparse_split as split, sparse_tensor_dense_matmul as matmul, sparse_tensor_to_dense as to_dense, sparse_to_indicator as to_indicator, sparse_transpose as transpose
from tensorflow.python.util import module_wrapper as _module_wrapper

"""Public API for tf._api.v2.sparse namespace
"""
if not isinstance(_sys.modules[__name__], _module_wrapper.TFModuleWrapper):
  ...
