"""
This type stub file was generated by pyright.
"""

import sys as _sys
from tensorflow.python.lib.io.file_io import copy as Copy, create_dir as MkDir, delete_file as Remove, delete_recursively as DeleteRecursively, file_exists as Exists, get_matching_files as Glob, is_directory as IsDirectory, list_directory as ListDirectory, recursive_create_dir as MakeDirs, rename as Rename, stat as Stat, walk as Walk
from tensorflow.python.platform.gfile import FastGFile, GFile
from tensorflow.python.util import module_wrapper as _module_wrapper

"""Public API for tf._api.v2.gfile namespace
"""
if not isinstance(_sys.modules[__name__], _module_wrapper.TFModuleWrapper):
  ...
