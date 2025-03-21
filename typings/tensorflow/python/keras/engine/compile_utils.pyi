"""
This type stub file was generated by pyright.
"""

"""Utilites for `Model.compile`."""
class Container:
  """Base Container class."""
  def __init__(self, output_names=...) -> None:
    ...
  
  def build(self, y_pred): # -> None:
    ...
  


class LossesContainer(Container):
  """A container class for losses passed to `Model.compile`."""
  def __init__(self, losses, loss_weights=..., output_names=...) -> None:
    ...
  
  @property
  def metrics(self): # -> list[Any] | list[Mean]:
    """Per-output loss metrics."""
    ...
  
  def build(self, y_pred): # -> None:
    """One-time setup of loss objects."""
    ...
  
  @property
  def built(self): # -> bool:
    ...
  
  def __call__(self, y_true, y_pred, sample_weight=..., regularization_losses=...): # -> defaultdict[Any, Any] | Any | list[Any] | object | SymbolicTensor | None:
    """Computes the overall loss.

    Args:
      y_true: An arbitrary structure of Tensors representing the ground truth.
      y_pred: An arbitrary structure of Tensors representing a Model's outputs.
      sample_weight: An arbitrary structure of Tensors representing the
        per-sample loss weights. If one Tensor is passed, it is used for all
        losses. If multiple Tensors are passed, the structure should match
        `y_pred`.
      regularization_losses: Additional losses to be added to the total loss.

    Returns:
      Tuple of `(total_loss, per_output_loss_list)`
    """
    ...
  
  def reset_state(self): # -> None:
    """Resets the state of loss metrics."""
    ...
  


class MetricsContainer(Container):
  """A container class for metrics passed to `Model.compile`."""
  def __init__(self, metrics=..., weighted_metrics=..., output_names=..., from_serialized=...) -> None:
    """Initializes a container for metrics.

    Arguments:
      metrics: see the `metrics` argument from `tf.keras.Model.compile`.
      weighted_metrics: see the `weighted_metrics` argument from
        `tf.keras.Model.compile`.
      output_names: A list of strings of names of outputs for the model.
      from_serialized: Whether the model being compiled is from a serialized
        model.  Used to avoid redundantly applying pre-processing renaming
        steps.
    """
    ...
  
  @property
  def metrics(self): # -> list[Any]:
    """All metrics in this container."""
    ...
  
  @property
  def unweighted_metrics(self): # -> list[None] | object | None:
    """Metrics in this container that should not be passed `sample_weight`."""
    ...
  
  @property
  def weighted_metrics(self): # -> list[None] | object | None:
    """Metrics in this container that should be passed `sample_weight`."""
    ...
  
  def build(self, y_pred, y_true): # -> None:
    """One-time setup of metric objects."""
    ...
  
  @property
  def built(self): # -> bool:
    ...
  
  def update_state(self, y_true, y_pred, sample_weight=...): # -> None:
    """Updates the state of per-output metrics."""
    ...
  
  def reset_state(self): # -> None:
    """Resets the state of all `Metric`s in this container."""
    ...
  


def create_pseudo_output_names(outputs): # -> list[Any]:
  """Create pseudo output names for a subclassed Model."""
  ...

def create_pseudo_input_names(inputs): # -> list[Any]:
  """Create pseudo input names for a subclassed Model."""
  ...

def map_to_output_names(y_pred, output_names, struct): # -> list[Any]:
  """Maps a dict to a list using `output_names` as keys.

  This is a convenience feature only. When a `Model`'s outputs
  are a list, you can specify per-output losses and metrics as
  a dict, where the keys are the output names. If you specify
  per-output losses and metrics via the same structure as the
  `Model`'s outputs (recommended), no mapping is performed.

  For the Functional API, the output names are the names of the
  last layer of each output. For the Subclass API, the output names
  are determined by `create_pseudo_output_names` (For example:
  `['output_1', 'output_2']` for a list of outputs).

  This mapping preserves backwards compatibility for `compile` and
  `fit`.

  Args:
    y_pred: Sample outputs of the Model, to determine if this convenience
      feature should be applied (`struct` is returned unmodified if `y_pred`
      isn't a flat list).
    output_names: List. The names of the outputs of the Model.
    struct: The structure to map.

  Returns:
    `struct` mapped to a list in same order as `output_names`.
  """
  ...

def map_missing_dict_keys(y_pred, struct): # -> dict[Any, Any]:
  """Replaces missing dict keys in `struct` with `None` placeholders."""
  ...

def match_dtype_and_rank(y_t, y_p, sw): # -> tuple[Tensor | Any | SparseTensor | IndexedSlices | SymbolicTensor, Any, Tensor | Any | SparseTensor | IndexedSlices | SymbolicTensor]:
  """Match dtype and rank of predictions."""
  ...

def get_mask(y_p): # -> Any | None:
  """Returns Keras mask from tensor."""
  ...

def apply_mask(y_p, sw, mask): # -> Any | Tensor | SparseTensor | IndexedSlices | SymbolicTensor:
  """Applies any mask on predictions to sample weights."""
  ...

def get_custom_object_name(obj): # -> str | None:
  """Returns the name to use for a custom loss or metric callable.

  Args:
    obj: Custom loss of metric callable

  Returns:
    Name to use, or `None` if the object was not recognized.
  """
  ...

