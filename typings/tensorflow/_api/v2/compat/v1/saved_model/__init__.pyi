"""
This type stub file was generated by pyright.
"""

import sys as _sys
from tensorflow._api.v2.compat.v1.saved_model import builder, constants, experimental, loader, main_op, signature_constants, signature_def_utils, tag_constants, utils
from tensorflow.python.saved_model.builder_impl import SavedModelBuilder as Builder
from tensorflow.python.saved_model.constants import ASSETS_DIRECTORY, ASSETS_KEY, DEBUG_DIRECTORY, DEBUG_INFO_FILENAME_PB, LEGACY_INIT_OP_KEY, MAIN_OP_KEY, SAVED_MODEL_FILENAME_PB, SAVED_MODEL_FILENAME_PBTXT, SAVED_MODEL_SCHEMA_VERSION, VARIABLES_DIRECTORY, VARIABLES_FILENAME
from tensorflow.python.saved_model.load import load as load_v2
from tensorflow.python.saved_model.loader_impl import load, maybe_saved_model_directory as contains_saved_model
from tensorflow.python.saved_model.main_op_impl import main_op_with_restore
from tensorflow.python.saved_model.save import save
from tensorflow.python.saved_model.save_options import SaveOptions
from tensorflow.python.saved_model.signature_constants import CLASSIFY_INPUTS, CLASSIFY_METHOD_NAME, CLASSIFY_OUTPUT_CLASSES, CLASSIFY_OUTPUT_SCORES, DEFAULT_SERVING_SIGNATURE_DEF_KEY, PREDICT_INPUTS, PREDICT_METHOD_NAME, PREDICT_OUTPUTS, REGRESS_INPUTS, REGRESS_METHOD_NAME, REGRESS_OUTPUTS
from tensorflow.python.saved_model.signature_def_utils_impl import build_signature_def, classification_signature_def, is_valid_signature, predict_signature_def, regression_signature_def
from tensorflow.python.saved_model.simple_save import simple_save
from tensorflow.python.saved_model.tag_constants import GPU, SERVING, TPU, TRAINING
from tensorflow.python.saved_model.utils_impl import build_tensor_info, get_tensor_from_tensor_info
from tensorflow.python.trackable.asset import Asset
from tensorflow.python.util import module_wrapper as _module_wrapper

"""Public API for tf._api.v2.saved_model namespace
"""
if not isinstance(_sys.modules[__name__], _module_wrapper.TFModuleWrapper):
  ...
