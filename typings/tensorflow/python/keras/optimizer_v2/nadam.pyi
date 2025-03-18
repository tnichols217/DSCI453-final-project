"""
This type stub file was generated by pyright.
"""

from tensorflow.python.keras.optimizer_v2 import optimizer_v2

"""Nadam optimizer implementation."""
class Nadam(optimizer_v2.OptimizerV2):
  r"""Optimizer that implements the NAdam algorithm.
  Much like Adam is essentially RMSprop with momentum, Nadam is Adam with
  Nesterov momentum.

  Args:
    learning_rate: A Tensor or a floating point value.  The learning rate.
    beta_1: A float value or a constant float tensor. The exponential decay
      rate for the 1st moment estimates.
    beta_2: A float value or a constant float tensor. The exponential decay
      rate for the exponentially weighted infinity norm.
    epsilon: A small constant for numerical stability.
    name: Optional name for the operations created when applying gradients.
      Defaults to `"Nadam"`.
    **kwargs: Keyword arguments. Allowed to be one of
      `"clipnorm"` or `"clipvalue"`.
      `"clipnorm"` (float) clips gradients by norm; `"clipvalue"` (float) clips
      gradients by value.

  Usage Example:
    >>> opt = tf.keras.optimizers.Nadam(learning_rate=0.2)
    >>> var1 = tf.Variable(10.0)
    >>> loss = lambda: (var1 ** 2) / 2.0
    >>> step_count = opt.minimize(loss, [var1]).numpy()
    >>> "{:.1f}".format(var1.numpy())
    9.8

  Reference:
    - [Dozat, 2015](http://cs229.stanford.edu/proj2015/054_report.pdf).
  """
  _HAS_AGGREGATE_GRAD = ...
  def __init__(self, learning_rate=..., beta_1=..., beta_2=..., epsilon=..., name=..., **kwargs) -> None:
    ...
  
  def get_config(self): # -> dict[str, str | Any]:
    ...
  


