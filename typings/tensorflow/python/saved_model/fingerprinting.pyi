"""
This type stub file was generated by pyright.
"""

from typing import Any
from tensorflow.core.protobuf import fingerprint_pb2
from tensorflow.python.saved_model.pywrap_saved_model import fingerprinting as fingerprinting_pywrap
from tensorflow.python.util.tf_export import tf_export

"""Methods for SavedModel fingerprinting.

This module contains classes and functions for reading the SavedModel
fingerprint.
"""
@tf_export("saved_model.experimental.Fingerprint", v1=[])
class Fingerprint:
  """The SavedModel fingerprint.

  Each attribute of this class is named after a field name in the
  FingerprintDef proto and contains the value of the respective field in the
  protobuf.

  Attributes:
    saved_model_checksum: A uint64 containing the `saved_model_checksum`.
    graph_def_program_hash: A uint64 containing `graph_def_program_hash`.
    signature_def_hash: A uint64 containing the `signature_def_hash`.
    saved_object_graph_hash: A uint64 containing the `saved_object_graph_hash`.
    checkpoint_hash: A uint64 containing the`checkpoint_hash`.
    version: An int32 containing the producer field of the VersionDef.
  """
  def __init__(self, saved_model_checksum: int = ..., graph_def_program_hash: int = ..., signature_def_hash: int = ..., saved_object_graph_hash: int = ..., checkpoint_hash: int = ..., version: int = ...) -> None:
    """Initializes the instance based on values in the SavedModel fingerprint.

    Args:
      saved_model_checksum: Value of the`saved_model_checksum`.
      graph_def_program_hash: Value of the `graph_def_program_hash`.
      signature_def_hash: Value of the `signature_def_hash`.
      saved_object_graph_hash: Value of the `saved_object_graph_hash`.
      checkpoint_hash: Value of the `checkpoint_hash`.
      version: Value of the producer field of the VersionDef.
    """
    ...
  
  @classmethod
  def from_proto(cls, proto: fingerprint_pb2.FingerprintDef) -> Fingerprint:
    """Constructs Fingerprint object from protocol buffer message."""
    ...
  
  def __eq__(self, other: Any) -> bool:
    ...
  
  def __str__(self) -> str:
    ...
  
  def __repr__(self) -> str:
    ...
  
  def singleprint(self) -> fingerprinting_pywrap.Singleprint:
    """Canonical fingerprinting ID for a SavedModel.

    Uniquely identifies a SavedModel based on the regularized fingerprint
    attributes. (saved_model_checksum is sensitive to immaterial changes and
    thus non-deterministic.)

    Returns:
      The string concatenation of `graph_def_program_hash`,
      `signature_def_hash`, `saved_object_graph_hash`, and `checkpoint_hash`
      fingerprint attributes (separated by '/').

    Raises:
      ValueError: If the fingerprint fields cannot be used to construct the
      singleprint.
    """
    ...
  


@tf_export("saved_model.experimental.read_fingerprint", v1=[])
def read_fingerprint(export_dir: str) -> Fingerprint:
  """Reads the fingerprint of a SavedModel in `export_dir`.

  Returns a `tf.saved_model.experimental.Fingerprint` object that contains
  the values of the SavedModel fingerprint, which is persisted on disk in the
  `fingerprint.pb` file in the `export_dir`.

  Read more about fingerprints in the SavedModel guide at
  https://www.tensorflow.org/guide/saved_model.

  Args:
    export_dir: The directory that contains the SavedModel.

  Returns:
    A `tf.saved_model.experimental.Fingerprint`.

  Raises:
    FileNotFoundError: If no or an invalid fingerprint is found.
  """
  ...

