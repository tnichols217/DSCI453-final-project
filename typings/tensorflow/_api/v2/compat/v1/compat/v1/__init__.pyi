"""
This type stub file was generated by pyright.
"""

import os as _os
import sys as _sys
from tensorflow.python.tools import module_util as _module_util
from tensorflow.python.util.lazy_loader import KerasLazyLoader as _KerasLazyLoader
from tensorflow._api.v2.compat.v1 import __internal__, app, audio, autograph, bitwise, compat, config, data, debugging, distribute, distributions, dtypes, errors, experimental, feature_column, gfile, graph_util, image, initializers, io, layers, linalg, lite, logging, lookup, losses, manip, math, metrics, mixed_precision, mlir, nest, nn, profiler, python_io, quantization, queue, ragged, random, raw_ops, resource_loader, saved_model, sets, signal, sparse, spectral, strings, summary, sysconfig, test, tpu, train, types, user_ops, version, xla
from tensorflow.python.ops.gen_array_ops import batch_to_space_nd, bitcast, broadcast_to, check_numerics, diag, extract_volume_patches, fake_quant_with_min_max_args, fake_quant_with_min_max_args_gradient, fake_quant_with_min_max_vars, fake_quant_with_min_max_vars_gradient, fake_quant_with_min_max_vars_per_channel, fake_quant_with_min_max_vars_per_channel_gradient, identity_n, invert_permutation, matrix_band_part, quantized_concat, reverse_v2 as reverse, scatter_nd, space_to_batch_nd, tensor_scatter_add, tensor_scatter_max as tensor_scatter_nd_max, tensor_scatter_min as tensor_scatter_nd_min, tensor_scatter_sub as tensor_scatter_nd_sub, tile, unravel_index
from tensorflow.python.ops.gen_control_flow_ops import no_op
from tensorflow.python.ops.gen_data_flow_ops import dynamic_partition, dynamic_stitch
from tensorflow.python.ops.gen_experimental_dataset_ops import check_pinned
from tensorflow.python.ops.gen_io_ops import matching_files, write_file
from tensorflow.python.ops.gen_linalg_ops import cholesky, matrix_determinant, matrix_inverse, matrix_solve, matrix_square_root, qr
from tensorflow.python.ops.gen_logging_ops import timestamp
from tensorflow.python.ops.gen_math_ops import acosh, asin, asinh, atan, atan2, atanh, betainc, cos, cosh, cross, digamma, erf, erfc, expm1, floor_div, floor_mod as floormod, greater, greater_equal, igamma, igammac, is_finite, is_inf, is_nan, less, less_equal, lgamma, log, log1p, logical_and, logical_not, logical_or, maximum, minimum, neg as negative, polygamma, real_div as realdiv, reciprocal, rint, segment_max, segment_mean, segment_min, segment_prod, segment_sum, sin, sinh, square, squared_difference, tan, tanh, truncate_div as truncatediv, truncate_mod as truncatemod, unsorted_segment_max, unsorted_segment_min, unsorted_segment_prod, unsorted_segment_sum, zeta
from tensorflow.python.ops.gen_nn_ops import approx_top_k, conv, conv2d_backprop_filter_v2, conv2d_backprop_input_v2
from tensorflow.python.ops.gen_parsing_ops import decode_compressed, parse_tensor
from tensorflow.python.ops.gen_ragged_array_ops import ragged_fill_empty_rows, ragged_fill_empty_rows_grad
from tensorflow.python.ops.gen_random_index_shuffle_ops import random_index_shuffle
from tensorflow.python.ops.gen_spectral_ops import fft, fft2d, fft3d, fftnd, ifft, ifft2d, ifft3d, ifftnd, irfftnd, rfftnd
from tensorflow.python.ops.gen_string_ops import as_string, decode_base64, encode_base64, string_strip, string_to_hash_bucket_fast, string_to_hash_bucket_strong
from tensorflow.python.client.session import InteractiveSession, Session
from tensorflow.python.compat.v2_compat import disable_v2_behavior, enable_v2_behavior
from tensorflow.python.data.ops.optional_ops import OptionalSpec
from tensorflow.python.eager.backprop import GradientTape
from tensorflow.python.eager.context import executing_eagerly_v1 as executing_eagerly
from tensorflow.python.eager.polymorphic_function.polymorphic_function import function
from tensorflow.python.eager.wrap_function import wrap_function
from tensorflow.python.framework.constant_op import constant_v1 as constant
from tensorflow.python.framework.device_spec import DeviceSpecV1 as DeviceSpec
from tensorflow.python.framework.dtypes import DType, QUANTIZED_DTYPES, as_dtype, bfloat16, bool, complex128, complex64, double, float16, float32, float64, half, int16, int32, int64, int8, qint16, qint32, qint8, quint16, quint8, resource, string, uint16, uint32, uint64, uint8, variant
from tensorflow.python.framework.errors_impl import OpError
from tensorflow.python.framework.graph_util_impl import GraphDef
from tensorflow.python.framework.importer import import_graph_def
from tensorflow.python.framework.indexed_slices import IndexedSlices, IndexedSlicesSpec, convert_to_tensor_or_indexed_slices
from tensorflow.python.framework.load_library import load_file_system_library, load_library, load_op_library
from tensorflow.python.framework.ops import Graph, GraphKeys, Operation, RegisterGradient, _colocate_with as colocate_with, add_to_collection, add_to_collections, container, control_dependencies, device, disable_eager_execution, enable_eager_execution, executing_eagerly_outside_functions, get_collection, get_collection_ref, get_default_graph, init_scope, is_symbolic_tensor, name_scope_v1 as name_scope, no_gradient as NoGradient, op_scope, reset_default_graph
from tensorflow.python.framework.random_seed import get_seed, set_random_seed
from tensorflow.python.framework.sparse_tensor import SparseTensor, SparseTensorSpec, SparseTensorValue, convert_to_tensor_or_sparse_tensor
from tensorflow.python.framework.stack import get_default_session
from tensorflow.python.framework.tensor import Tensor, TensorSpec, disable_tensor_equality, enable_tensor_equality
from tensorflow.python.framework.tensor_conversion import convert_to_tensor_v1_with_dispatch as convert_to_tensor
from tensorflow.python.framework.tensor_conversion_registry import register_tensor_conversion_function
from tensorflow.python.framework.tensor_shape import Dimension, TensorShape, dimension_at_index, dimension_value, disable_v2_tensorshape, enable_v2_tensorshape
from tensorflow.python.framework.tensor_util import MakeNdarray as make_ndarray, constant_value as get_static_value, is_tf_type as is_tensor, make_tensor_proto
from tensorflow.python.framework.type_spec import TypeSpec, type_spec_from_value
from tensorflow.python.framework.versions import COMPILER_VERSION, CXX11_ABI_FLAG, CXX_VERSION, GIT_VERSION, GRAPH_DEF_VERSION, GRAPH_DEF_VERSION_MIN_CONSUMER, GRAPH_DEF_VERSION_MIN_PRODUCER, MONOLITHIC_BUILD, VERSION
from tensorflow.python.module.module import Module
from tensorflow.python.ops.array_ops import batch_gather, batch_to_space, boolean_mask, broadcast_dynamic_shape, broadcast_static_shape, concat, depth_to_space, dequantize, edit_distance, expand_dims, extract_image_patches, fill, fingerprint, gather, gather_nd, guarantee_const, identity, matrix_diag, matrix_diag_part, matrix_set_diag, matrix_transpose, meshgrid, newaxis, one_hot, ones, ones_like, pad, parallel_stack, placeholder, placeholder_with_default, quantize, quantize_v2, rank, repeat, required_space_to_batch_paddings, reshape, reverse_sequence, searchsorted, sequence_mask, setdiff1d, shape, shape_n, size, slice, space_to_batch, space_to_depth, sparse_mask, sparse_placeholder, split, squeeze, stop_gradient, strided_slice, tensor_diag_part as diag_part, tensor_scatter_nd_update, transpose, unique, unique_with_counts, where, where_v2, zeros, zeros_like
from tensorflow.python.ops.array_ops_stack import stack, unstack
from tensorflow.python.ops.batch_ops import batch_function as nondifferentiable_batch_function
from tensorflow.python.ops.bincount_ops import bincount_v1 as bincount
from tensorflow.python.ops.check_ops import assert_equal, assert_greater, assert_greater_equal, assert_integer, assert_less, assert_less_equal, assert_near, assert_negative, assert_non_negative, assert_non_positive, assert_none_equal, assert_positive, assert_proper_iterable, assert_rank, assert_rank_at_least, assert_rank_in, assert_same_float_dtype, assert_scalar, assert_type, ensure_shape, is_non_decreasing, is_numeric_tensor, is_strictly_increasing
from tensorflow.python.ops.clip_ops import clip_by_average_norm, clip_by_global_norm, clip_by_norm, clip_by_value, global_norm
from tensorflow.python.ops.cond import cond
from tensorflow.python.ops.confusion_matrix import confusion_matrix_v1 as confusion_matrix
from tensorflow.python.ops.control_flow_assert import Assert
from tensorflow.python.ops.control_flow_case import case
from tensorflow.python.ops.control_flow_ops import group, tuple
from tensorflow.python.ops.control_flow_switch_case import switch_case
from tensorflow.python.ops.control_flow_v2_toggles import control_flow_v2_enabled, disable_control_flow_v2, enable_control_flow_v2
from tensorflow.python.ops.critical_section_ops import CriticalSection
from tensorflow.python.ops.custom_gradient import custom_gradient, grad_pass_through, recompute_grad
from tensorflow.python.ops.data_flow_ops import ConditionalAccumulator, ConditionalAccumulatorBase, FIFOQueue, PaddingFIFOQueue, PriorityQueue, QueueBase, RandomShuffleQueue, SparseConditionalAccumulator
from tensorflow.python.ops.functional_ops import foldl, foldr, scan
from tensorflow.python.ops.gradients_impl import gradients, hessians
from tensorflow.python.ops.gradients_util import AggregationMethod
from tensorflow.python.ops.histogram_ops import histogram_fixed_width, histogram_fixed_width_bins
from tensorflow.python.ops.init_ops import Constant as constant_initializer, GlorotNormal as glorot_normal_initializer, GlorotUniform as glorot_uniform_initializer, Ones as ones_initializer, Orthogonal as orthogonal_initializer, RandomNormal as random_normal_initializer, RandomUniform as random_uniform_initializer, TruncatedNormal as truncated_normal_initializer, UniformUnitScaling as uniform_unit_scaling_initializer, VarianceScaling as variance_scaling_initializer, Zeros as zeros_initializer
from tensorflow.python.ops.io_ops import FixedLengthRecordReader, IdentityReader, LMDBReader, ReaderBase, TFRecordReader, TextLineReader, WholeFileReader, read_file, serialize_tensor
from tensorflow.python.ops.linalg_ops import cholesky_solve, eye, matrix_solve_ls, matrix_triangular_solve, norm, self_adjoint_eig, self_adjoint_eigvals, svd
from tensorflow.python.ops.logging_ops import Print, print_v2 as print
from tensorflow.python.ops.lookup_ops import initialize_all_tables, tables_initializer
from tensorflow.python.ops.manip_ops import roll
from tensorflow.python.ops.map_fn import map_fn
from tensorflow.python.ops.math_ops import abs, accumulate_n, acos, add, add_n, angle, arg_max, arg_min, argmax, argmin, cast, ceil, complex, conj, count_nonzero, cumprod, cumsum, div, div_no_nan, divide, equal, exp, floor, floordiv, imag, linspace_nd as lin_space, log_sigmoid, logical_xor, matmul, multiply, not_equal, pow, range, real, reduce_all_v1 as reduce_all, reduce_any_v1 as reduce_any, reduce_logsumexp_v1 as reduce_logsumexp, reduce_max_v1 as reduce_max, reduce_mean_v1 as reduce_mean, reduce_min_v1 as reduce_min, reduce_prod_v1 as reduce_prod, reduce_sum_v1 as reduce_sum, round, rsqrt, saturate_cast, scalar_mul, sigmoid, sign, sparse_matmul, sparse_segment_mean, sparse_segment_sqrt_n, sparse_segment_sum, sqrt, subtract, tensordot, to_bfloat16, to_complex128, to_complex64, to_double, to_float, to_int32, to_int64, trace, truediv, unsorted_segment_mean, unsorted_segment_sqrt_n
from tensorflow.python.ops.numerics import add_check_numerics_ops, verify_tensor_all_finite
from tensorflow.python.ops.parallel_for.control_flow_ops import vectorized_map
from tensorflow.python.ops.parsing_config import FixedLenFeature, FixedLenSequenceFeature, SparseFeature, VarLenFeature
from tensorflow.python.ops.parsing_ops import decode_csv, decode_json_example, decode_raw_v1 as decode_raw, parse_example, parse_single_example, parse_single_sequence_example
from tensorflow.python.ops.partitioned_variables import create_partitioned_variables, fixed_size_partitioner, min_max_variable_partitioner, variable_axis_size_partitioner
from tensorflow.python.ops.ragged.ragged_string_ops import string_split
from tensorflow.python.ops.ragged.ragged_tensor import RaggedTensor, RaggedTensorSpec
from tensorflow.python.ops.random_crop_ops import random_crop
from tensorflow.python.ops.random_ops import multinomial, random_gamma, random_normal, random_poisson, random_shuffle, random_uniform, truncated_normal
from tensorflow.python.ops.resource_variables_toggle import disable_resource_variables, enable_resource_variables, resource_variables_enabled
from tensorflow.python.ops.script_ops import eager_py_func as py_function, numpy_function, py_func
from tensorflow.python.ops.session_ops import delete_session_tensor, get_session_handle, get_session_tensor
from tensorflow.python.ops.sort_ops import argsort, sort
from tensorflow.python.ops.sparse_ops import deserialize_many_sparse, serialize_many_sparse, serialize_sparse, sparse_add, sparse_concat, sparse_fill_empty_rows, sparse_maximum, sparse_merge, sparse_minimum, sparse_reduce_max, sparse_reduce_max_sparse, sparse_reduce_sum, sparse_reduce_sum_sparse, sparse_reorder, sparse_reset_shape, sparse_reshape, sparse_retain, sparse_slice, sparse_softmax, sparse_split, sparse_tensor_dense_matmul, sparse_tensor_to_dense, sparse_to_dense, sparse_to_indicator, sparse_transpose
from tensorflow.python.ops.special_math_ops import einsum, lbeta
from tensorflow.python.ops.state_ops import assign, assign_add, assign_sub, batch_scatter_update, count_up_to, scatter_add, scatter_div, scatter_max, scatter_min, scatter_mul, scatter_nd_add, scatter_nd_sub, scatter_nd_update, scatter_sub, scatter_update
from tensorflow.python.ops.string_ops import reduce_join, regex_replace, string_join, string_to_hash_bucket_v1 as string_to_hash_bucket, string_to_number_v1 as string_to_number, substr_deprecated as substr
from tensorflow.python.ops.template import make_template
from tensorflow.python.ops.tensor_array_ops import TensorArray, TensorArraySpec
from tensorflow.python.ops.unconnected_gradients import UnconnectedGradients
from tensorflow.python.ops.variable_scope import AUTO_REUSE, VariableScope, get_local_variable, get_variable, get_variable_scope, no_regularizer, variable_creator_scope_v1 as variable_creator_scope, variable_op_scope, variable_scope
from tensorflow.python.ops.variable_v1 import VariableV1 as Variable, is_variable_initialized
from tensorflow.python.ops.variables import VariableAggregation, VariableSynchronization, all_variables, assert_variables_initialized, global_variables, global_variables_initializer, initialize_all_variables, initialize_local_variables, initialize_variables, local_variables, local_variables_initializer, model_variables, moving_average_variables, report_uninitialized_variables, trainable_variables, variables_initializer
from tensorflow.python.ops.while_loop import while_loop
from tensorflow.python.platform.tf_logging import get_logger
from tensorflow.python.proto_exports import AttrValue, ConfigProto, Event, GPUOptions, GraphOptions, HistogramProto, LogMessage, MetaGraphDef, NameAttrList, NodeDef, OptimizerOptions, RunMetadata, RunOptions, SessionLog, Summary, SummaryMetadata, TensorInfo
from tensorflow.python.util import module_wrapper as _module_wrapper
from tensorflow.python.platform import flags

"""Bring in all of the public TensorFlow interface into this module."""
if not isinstance(_sys.modules[__name__], _module_wrapper.TFModuleWrapper):
  ...
_current_module = ...
_tf_uses_legacy_keras = ...
if _tf_uses_legacy_keras:
  _module_dir = ...
else:
  _module_dir = ...
if _tf_uses_legacy_keras:
  _module_dir = ...
else:
  _module_dir = ...
if _tf_uses_legacy_keras:
  _module_dir = ...
else:
  _module_dir = ...
