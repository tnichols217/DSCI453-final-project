"""
This type stub file was generated by pyright.
"""

"""Utils related to keras model saving."""
def extract_model_metrics(model): # -> dict[Any, Any] | None:
  """Convert metrics from a Keras model `compile` API to dictionary.

  This is used for converting Keras models to SavedModels.

  Args:
    model: A `tf.keras.Model` object.

  Returns:
    Dictionary mapping metric names to metric instances. May return `None` if
    the model does not contain any metrics.
  """
  ...

def model_input_signature(model, keep_original_batch_size=...): # -> defaultdict[Any, Any] | Any | list[Any] | object | list[Any | defaultdict[Any, Any] | list[Any] | object | None] | None:
  """Inspect model to get its input signature.

  The model's input signature is a list with a single (possibly-nested) object.
  This is due to the Keras-enforced restriction that tensor inputs must be
  passed in as the first argument.

  For example, a model with input {'feature1': <Tensor>, 'feature2': <Tensor>}
  will have input signature: [{'feature1': TensorSpec, 'feature2': TensorSpec}]

  Args:
    model: Keras Model object.
    keep_original_batch_size: A boolean indicating whether we want to keep using
      the original batch size or set it to None. Default is `False`, which means
      that the batch dim of the returned input signature will always be set to
      `None`.

  Returns:
    A list containing either a single TensorSpec or an object with nested
    TensorSpecs. This list does not contain the `training` argument.
  """
  ...

def raise_model_input_error(model):
  ...

def trace_model_call(model, input_signature=...): # -> None:
  """Trace the model call to create a tf.function for exporting a Keras model.

  Args:
    model: A Keras model.
    input_signature: optional, a list of tf.TensorSpec objects specifying the
      inputs to the model.

  Returns:
    A tf.function wrapping the model's call function with input signatures set.

  Raises:
    ValueError: if input signature cannot be inferred from the model.
  """
  ...

def model_metadata(model, include_optimizer=..., require_config=...): # -> dict[str, str | dict[str, Any]]:
  """Returns a dictionary containing the model metadata."""
  ...

def should_overwrite(filepath, overwrite): # -> bool:
  """Returns whether the filepath should be overwritten."""
  ...

def compile_args_from_training_config(training_config, custom_objects=...): # -> dict[str, Any | None]:
  """Return model.compile arguments from training config."""
  ...

def try_build_compiled_arguments(model): # -> None:
  ...

def is_hdf5_filepath(filepath):
  ...

