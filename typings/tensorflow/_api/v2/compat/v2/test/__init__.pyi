"""
This type stub file was generated by pyright.
"""

import sys as _sys
from tensorflow._api.v2.compat.v2.test import experimental
from tensorflow.python.framework.test_util import TensorFlowTestCase as TestCase, assert_equal_graph_def_v2 as assert_equal_graph_def, create_local_cluster, gpu_device_name, is_gpu_available, with_eager_op_as_function
from tensorflow.python.ops.gradient_checker_v2 import compute_gradient
from tensorflow.python.platform.benchmark import TensorFlowBenchmark as Benchmark, benchmark_config
from tensorflow.python.platform.test import disable_with_predicate, is_built_with_cuda, is_built_with_gpu_support, is_built_with_rocm, is_built_with_xla, is_cpu_target_available, main

"""Public API for tf._api.v2.test namespace
"""
