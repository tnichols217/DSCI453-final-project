"""
This type stub file was generated by pyright.
"""

from tensorflow.python.util.tf_export import tf_export

"""Contains global variables related to mixed precision.

This is not part of mixed_precision.py to avoid a circular dependency.
mixed_precision.py depends on Session, and Session depends on this file.
"""
_mixed_precision_graph_rewrite_is_enabled = ...
_non_mixed_precision_session_created = ...
_using_mixed_precision_policy = ...
@tf_export('__internal__.train.is_mixed_precision_graph_rewrite_enabled', v1=[])
def is_mixed_precision_graph_rewrite_enabled(): # -> bool:
  ...

def set_mixed_precision_graph_rewrite_enabled(enabled): # -> None:
  ...

def non_mixed_precision_session_created(): # -> bool:
  ...

def set_non_mixed_precision_session_created(created): # -> None:
  ...

def is_using_mixed_precision_policy(): # -> bool:
  ...

@tf_export('__internal__.train.set_using_mixed_precision_policy', v1=[])
def set_using_mixed_precision_policy(is_using): # -> None:
  ...

