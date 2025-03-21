"""
This type stub file was generated by pyright.
"""

import sys as _sys
from tensorflow.python.ops.summary_ops_v2 import all_v2_summary_ops, initialize
from tensorflow.python.proto_exports import Event, SessionLog, Summary, SummaryDescription, TaggedRunMetadata
from tensorflow.python.summary.summary import audio, get_summary_description, histogram, image, merge, merge_all, scalar, tensor_summary, text
from tensorflow.python.summary.writer.writer import FileWriter
from tensorflow.python.summary.writer.writer_cache import FileWriterCache
from tensorflow.python.util import module_wrapper as _module_wrapper

"""Public API for tf._api.v2.summary namespace
"""
if not isinstance(_sys.modules[__name__], _module_wrapper.TFModuleWrapper):
  ...
