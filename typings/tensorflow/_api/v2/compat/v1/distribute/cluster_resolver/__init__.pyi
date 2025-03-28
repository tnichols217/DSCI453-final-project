"""
This type stub file was generated by pyright.
"""

import sys as _sys
from tensorflow.python.distribute.cluster_resolver.cluster_resolver import ClusterResolver, SimpleClusterResolver, UnionClusterResolver as UnionResolver
from tensorflow.python.distribute.cluster_resolver.gce_cluster_resolver import GCEClusterResolver
from tensorflow.python.distribute.cluster_resolver.kubernetes_cluster_resolver import KubernetesClusterResolver
from tensorflow.python.distribute.cluster_resolver.slurm_cluster_resolver import SlurmClusterResolver
from tensorflow.python.distribute.cluster_resolver.tfconfig_cluster_resolver import TFConfigClusterResolver
from tensorflow.python.distribute.cluster_resolver.tpu_cluster_resolver import TPUClusterResolver
from tensorflow.python.util import module_wrapper as _module_wrapper

"""Public API for tf._api.v2.distribute.cluster_resolver namespace
"""
if not isinstance(_sys.modules[__name__], _module_wrapper.TFModuleWrapper):
  ...
