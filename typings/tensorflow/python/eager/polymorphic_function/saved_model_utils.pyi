"""
This type stub file was generated by pyright.
"""

from tensorflow.python.saved_model import registration
from tensorflow.python.trackable import base as trackable

"""A shim layer for working with functions exported/restored from saved models.

This functionality should ultimately be moved into a first-class core API.
"""
@registration.register_tf_serializable()
class TrackableConstant(trackable.Trackable):
  """Trackable class for captured constants."""
  __slots__ = ...
  def __init__(self, capture, function) -> None:
    ...
  


