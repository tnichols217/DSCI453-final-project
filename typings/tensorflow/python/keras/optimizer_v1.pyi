"""
This type stub file was generated by pyright.
"""

from tensorflow.python.trackable import base as trackable

"""Legacy v1 optimizer classes.

For more examples see the base class `tf.compat.v1.keras.optimizers.Optimizer`.
"""
class Optimizer:
  """Abstract optimizer base class.

  Note: this is the parent class of all optimizers, not an actual optimizer
  that can be used for training models.

  All Keras optimizers support the following keyword arguments:

      clipnorm: float >= 0. Gradients will be clipped
          when their L2 norm exceeds this value.
      clipvalue: float >= 0. Gradients will be clipped
          when their absolute value exceeds this value.
  """
  def __init__(self, **kwargs) -> None:
    ...
  
  _HAS_AGGREGATE_GRAD = ...
  def get_updates(self, loss, params):
    ...
  
  def get_gradients(self, loss, params): # -> list[IndexedSlices | Any] | list[IndexedSlices | Any | defaultdict[Any, Any] | list[Any] | object | None]:
    """Returns gradients of `loss` with respect to `params`.

    Args:
        loss: Loss tensor.
        params: List of variables.

    Returns:
        List of gradient tensors.

    Raises:
        ValueError: In case any gradient cannot be computed (e.g. if gradient
          function not implemented).
    """
    ...
  
  def set_weights(self, weights): # -> None:
    """Sets the weights of the optimizer, from Numpy arrays.

    Should only be called after computing the gradients
    (otherwise the optimizer has no weights).

    Args:
        weights: a list of Numpy arrays. The number of arrays and their shape
          must match number of the dimensions of the weights of the optimizer
          (i.e. it should match the output of `get_weights`).

    Raises:
        ValueError: in case of incompatible weight shapes.
    """
    ...
  
  def get_weights(self): # -> list[Any] | tuple[Any, ...] | Any | defaultdict[Any, Any] | None:
    """Returns the current value of the weights of the optimizer.

    Returns:
        A list of numpy arrays.
    """
    ...
  
  def get_config(self): # -> dict[Any, Any]:
    ...
  
  @classmethod
  def from_config(cls, config): # -> Self:
    ...
  


class SGD(Optimizer):
  """Stochastic gradient descent optimizer.

  Includes support for momentum,
  learning rate decay, and Nesterov momentum.

  Args:
      lr: float >= 0. Learning rate.
      momentum: float >= 0. Parameter that accelerates SGD in the relevant
        direction and dampens oscillations.
      decay: float >= 0. Learning rate decay over each update.
      nesterov: boolean. Whether to apply Nesterov momentum.
  """
  def __init__(self, lr=..., momentum=..., decay=..., nesterov=..., **kwargs) -> None:
    ...
  
  def get_updates(self, loss, params): # -> list[Any]:
    ...
  
  def get_config(self): # -> dict[Any, Any]:
    ...
  


class RMSprop(Optimizer):
  """RMSProp optimizer.

  It is recommended to leave the parameters of this optimizer
  at their default values
  (except the learning rate, which can be freely tuned).

  Args:
    lr: float >= 0. Learning rate.
    rho: float >= 0.
    epsilon: float >= 0. Fuzz factor.
      If `None`, defaults to `backend.epsilon()`.
    decay: float >= 0. Learning rate decay over each update.
  """
  def __init__(self, lr=..., rho=..., epsilon=..., decay=..., **kwargs) -> None:
    ...
  
  def get_updates(self, loss, params): # -> list[Any]:
    ...
  
  def get_config(self): # -> dict[Any, Any]:
    ...
  


class Adagrad(Optimizer):
  """Adagrad optimizer.

  Adagrad is an optimizer with parameter-specific learning rates,
  which are adapted relative to how frequently a parameter gets
  updated during training. The more updates a parameter receives,
  the smaller the updates.

  It is recommended to leave the parameters of this optimizer
  at their default values.

  # Arguments
      lr: float >= 0. Initial learning rate.
      epsilon: float >= 0. If `None`, defaults to `backend.epsilon()`.
      decay: float >= 0. Learning rate decay over each update.

  # References
      - [Adaptive Subgradient Methods for Online Learning and Stochastic
      Optimization](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
  """
  def __init__(self, lr=..., epsilon=..., decay=..., **kwargs) -> None:
    ...
  
  def get_updates(self, loss, params): # -> list[Any]:
    ...
  
  def get_config(self): # -> dict[Any, Any]:
    ...
  


class Adadelta(Optimizer):
  """Adadelta optimizer.

  Adadelta is a more robust extension of Adagrad
  that adapts learning rates based on a moving window of gradient updates,
  instead of accumulating all past gradients. This way, Adadelta continues
  learning even when many updates have been done. Compared to Adagrad, in the
  original version of Adadelta you don't have to set an initial learning
  rate. In this version, initial learning rate and decay factor can
  be set, as in most other Keras optimizers.

  It is recommended to leave the parameters of this optimizer
  at their default values.

  Arguments:
    lr: float >= 0. Initial learning rate, defaults to 1.
        It is recommended to leave it at the default value.
    rho: float >= 0. Adadelta decay factor, corresponding to fraction of
        gradient to keep at each time step.
    epsilon: float >= 0. Fuzz factor.
      If `None`, defaults to `backend.epsilon()`.
    decay: float >= 0. Initial learning rate decay.

  References:
      - [Adadelta - an adaptive learning rate
      method](http://arxiv.org/abs/1212.5701)
  """
  def __init__(self, lr=..., rho=..., epsilon=..., decay=..., **kwargs) -> None:
    ...
  
  def get_updates(self, loss, params): # -> list[Any]:
    ...
  
  def get_config(self): # -> dict[Any, Any]:
    ...
  


class Adam(Optimizer):
  """Adam optimizer.

  Default parameters follow those provided in the original paper.

  Args:
    lr: float >= 0. Learning rate.
    beta_1: float, 0 < beta < 1. Generally close to 1.
    beta_2: float, 0 < beta < 1. Generally close to 1.
    epsilon: float >= 0. Fuzz factor.
      If `None`, defaults to `backend.epsilon()`.
    decay: float >= 0. Learning rate decay over each update.
    amsgrad: boolean. Whether to apply the AMSGrad variant of this algorithm
      from the paper "On the Convergence of Adam and Beyond".
  """
  def __init__(self, lr=..., beta_1=..., beta_2=..., epsilon=..., decay=..., amsgrad=..., **kwargs) -> None:
    ...
  
  def get_updates(self, loss, params): # -> list[Any]:
    ...
  
  def get_config(self): # -> dict[Any, Any]:
    ...
  


class Adamax(Optimizer):
  """Adamax optimizer from Adam paper's Section 7.

  It is a variant of Adam based on the infinity norm.
  Default parameters follow those provided in the paper.

  Args:
    lr: float >= 0. Learning rate.
    beta_1/beta_2: floats, 0 < beta < 1. Generally close to 1.
    epsilon: float >= 0. Fuzz factor.
      If `None`, defaults to `backend.epsilon()`.
    decay: float >= 0. Learning rate decay over each update.
  """
  def __init__(self, lr=..., beta_1=..., beta_2=..., epsilon=..., decay=..., **kwargs) -> None:
    ...
  
  def get_updates(self, loss, params): # -> list[Any]:
    ...
  
  def get_config(self): # -> dict[Any, Any]:
    ...
  


class Nadam(Optimizer):
  """Nesterov Adam optimizer.

  Much like Adam is essentially RMSprop with momentum,
  Nadam is Adam RMSprop with Nesterov momentum.

  Default parameters follow those provided in the paper.
  It is recommended to leave the parameters of this optimizer
  at their default values.

  Args:
    lr: float >= 0. Learning rate.
    beta_1/beta_2: floats, 0 < beta < 1. Generally close to 1.
    epsilon: float >= 0. Fuzz factor.
      If `None`, defaults to `backend.epsilon()`.
  """
  def __init__(self, lr=..., beta_1=..., beta_2=..., epsilon=..., schedule_decay=..., **kwargs) -> None:
    ...
  
  def get_updates(self, loss, params): # -> list[Any]:
    ...
  
  def get_config(self): # -> dict[Any, Any]:
    ...
  


class TFOptimizer(Optimizer, trackable.Trackable):
  """Wrapper class for native TensorFlow optimizers."""
  def __init__(self, optimizer, iterations=...) -> None:
    ...
  
  def minimize(self, loss, var_list, grad_loss=..., tape=...): # -> None:
    """Mimics the `OptimizerV2.minimize` API."""
    ...
  
  def apply_gradients(self, grads_and_vars): # -> None:
    ...
  
  def get_grads(self, loss, params):
    ...
  
  def get_updates(self, loss, params): # -> list[Any]:
    ...
  
  @property
  def weights(self):
    ...
  
  def get_config(self):
    ...
  
  def from_config(self, config):
    ...
  


sgd = SGD
rmsprop = RMSprop
adagrad = Adagrad
adadelta = Adadelta
adam = Adam
adamax = Adamax
nadam = Nadam
