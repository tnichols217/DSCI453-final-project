"""
This type stub file was generated by pyright.
"""

import threading
from tensorflow.python import tf2
from tensorflow.python.keras.initializers import initializers_v1, initializers_v2
from tensorflow.python.keras.utils import generic_utils, tf_inspect as inspect
from tensorflow.python.ops import init_ops

"""Keras initializer serialization / deserialization."""
LOCAL = ...
def populate_deserializable_objects(): # -> None:
  """Populates dict ALL_OBJECTS with every built-in initializer.
  """
  ...

def serialize(initializer): # -> Any | dict[str, Any] | None:
  ...

def deserialize(config, custom_objects=...): # -> Any | None:
  """Return an `Initializer` object from its config."""
  ...

def get(identifier): # -> Any | object | Callable[..., object] | None:
  """Retrieve a Keras initializer by the identifier.

  The `identifier` may be the string name of a initializers function or class (
  case-sensitively).

  >>> identifier = 'Ones'
  >>> tf.keras.initializers.deserialize(identifier)
  <...keras.initializers.initializers_v2.Ones...>

  You can also specify `config` of the initializer to this function by passing
  dict containing `class_name` and `config` as an identifier. Also note that the
  `class_name` must map to a `Initializer` class.

  >>> cfg = {'class_name': 'Ones', 'config': {}}
  >>> tf.keras.initializers.deserialize(cfg)
  <...keras.initializers.initializers_v2.Ones...>

  In the case that the `identifier` is a class, this method will return a new
  instance of the class by its constructor.

  Args:
    identifier: String or dict that contains the initializer name or
      configurations.

  Returns:
    Initializer instance base on the input identifier.

  Raises:
    ValueError: If the input identifier is not a supported type or in a bad
      format.
  """
  ...

