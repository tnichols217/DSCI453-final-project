"""
This type stub file was generated by pyright.
"""

from tensorflow.python.keras.optimizer_v2 import optimizer_v2

"""RMSprop optimizer implementation."""
class RMSprop(optimizer_v2.OptimizerV2):
  r"""Optimizer that implements the RMSprop algorithm.

  The gist of RMSprop is to:

  - Maintain a moving (discounted) average of the square of gradients
  - Divide the gradient by the root of this average

  This implementation of RMSprop uses plain momentum, not Nesterov momentum.

  The centered version additionally maintains a moving average of the
  gradients, and uses that average to estimate the variance.

  Args:
    learning_rate: A `Tensor`, floating point value, or a schedule that is a
      `tf.keras.optimizers.schedules.LearningRateSchedule`, or a callable
      that takes no arguments and returns the actual value to use. The
      learning rate. Defaults to 0.001.
    rho: Discounting factor for the history/coming gradient. Defaults to 0.9.
    momentum: A scalar or a scalar `Tensor`. Defaults to 0.0.
    epsilon: A small constant for numerical stability. This epsilon is
      "epsilon hat" in the Kingma and Ba paper (in the formula just before
      Section 2.1), not the epsilon in Algorithm 1 of the paper. Defaults to
      1e-7.
    centered: Boolean. If `True`, gradients are normalized by the estimated
      variance of the gradient; if False, by the uncentered second moment.
      Setting this to `True` may help with training, but is slightly more
      expensive in terms of computation and memory. Defaults to `False`.
    name: Optional name prefix for the operations created when applying
      gradients. Defaults to `"RMSprop"`.
    **kwargs: Keyword arguments. Allowed to be one of
      `"clipnorm"` or `"clipvalue"`.
      `"clipnorm"` (float) clips gradients by norm; `"clipvalue"` (float) clips
      gradients by value.

  Note that in the dense implementation of this algorithm, variables and their
  corresponding accumulators (momentum, gradient moving average, square
  gradient moving average) will be updated even if the gradient is zero
  (i.e. accumulators will decay, momentum will be applied). The sparse
  implementation (used when the gradient is an `IndexedSlices` object,
  typically because of `tf.gather` or an embedding lookup in the forward pass)
  will not update variable slices or their accumulators unless those slices
  were used in the forward pass (nor is there an "eventual" correction to
  account for these omitted updates). This leads to more efficient updates for
  large embedding lookup tables (where most of the slices are not accessed in
  a particular graph execution), but differs from the published algorithm.

  Usage:

  >>> opt = tf.keras.optimizers.RMSprop(learning_rate=0.1)
  >>> var1 = tf.Variable(10.0)
  >>> loss = lambda: (var1 ** 2) / 2.0    # d(loss) / d(var1) = var1
  >>> step_count = opt.minimize(loss, [var1]).numpy()
  >>> var1.numpy()
  9.683772

  Reference:
    - [Hinton, 2012](
      http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
  """
  _HAS_AGGREGATE_GRAD = ...
  def __init__(self, learning_rate=..., rho=..., momentum=..., epsilon=..., centered=..., name=..., **kwargs) -> None:
    """Construct a new RMSprop optimizer.

    Args:
      learning_rate: A `Tensor`, floating point value, or a schedule that is a
        `tf.keras.optimizers.schedules.LearningRateSchedule`, or a callable
        that takes no arguments and returns the actual value to use. The
        learning rate. Defaults to 0.001.
      rho: Discounting factor for the history/coming gradient. Defaults to 0.9.
      momentum: A scalar or a scalar `Tensor`. Defaults to 0.0.
      epsilon: A small constant for numerical stability. This epsilon is
        "epsilon hat" in the Kingma and Ba paper (in the formula just before
        Section 2.1), not the epsilon in Algorithm 1 of the paper. Defaults to
        1e-7.
      centered: Boolean. If `True`, gradients are normalized by the estimated
        variance of the gradient; if False, by the uncentered second moment.
        Setting this to `True` may help with training, but is slightly more
        expensive in terms of computation and memory. Defaults to `False`.
      name: Optional name prefix for the operations created when applying
        gradients. Defaults to "RMSprop".
      **kwargs: keyword arguments. Allowed to be {`clipnorm`, `clipvalue`, `lr`,
        `decay`}. `clipnorm` is clip gradients by norm; `clipvalue` is clip
        gradients by value, `decay` is included for backward compatibility to
        allow time inverse decay of learning rate. `lr` is included for backward
        compatibility, recommended to use `learning_rate` instead.

    @compatibility(eager)
    When eager execution is enabled, `learning_rate`, `decay`, `momentum`, and
    `epsilon` can each be a callable that takes no arguments and returns the
    actual value to use. This can be useful for changing these values across
    different invocations of optimizer functions.
    @end_compatibility
    """
    ...
  
  def set_weights(self, weights): # -> None:
    ...
  
  def get_config(self): # -> dict[str, str | Any]:
    ...
  


RMSProp = RMSprop
