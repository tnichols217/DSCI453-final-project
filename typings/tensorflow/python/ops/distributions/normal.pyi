"""
This type stub file was generated by pyright.
"""

from tensorflow.python.ops.distributions import distribution
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export

"""The Normal (Gaussian) distribution class."""
__all__ = ["Normal", "NormalWithSoftplusScale"]
@tf_export(v1=["distributions.Normal"])
class Normal(distribution.Distribution):
  """The Normal distribution with location `loc` and `scale` parameters.

  #### Mathematical details

  The probability density function (pdf) is,

  ```none
  pdf(x; mu, sigma) = exp(-0.5 (x - mu)**2 / sigma**2) / Z
  Z = (2 pi sigma**2)**0.5
  ```

  where `loc = mu` is the mean, `scale = sigma` is the std. deviation, and, `Z`
  is the normalization constant.

  The Normal distribution is a member of the [location-scale family](
  https://en.wikipedia.org/wiki/Location-scale_family), i.e., it can be
  constructed as,

  ```none
  X ~ Normal(loc=0, scale=1)
  Y = loc + scale * X
  ```

  #### Examples

  Examples of initialization of one or a batch of distributions.

  ```python
  import tensorflow_probability as tfp
  tfd = tfp.distributions

  # Define a single scalar Normal distribution.
  dist = tfd.Normal(loc=0., scale=3.)

  # Evaluate the cdf at 1, returning a scalar.
  dist.cdf(1.)

  # Define a batch of two scalar valued Normals.
  # The first has mean 1 and standard deviation 11, the second 2 and 22.
  dist = tfd.Normal(loc=[1, 2.], scale=[11, 22.])

  # Evaluate the pdf of the first distribution on 0, and the second on 1.5,
  # returning a length two tensor.
  dist.prob([0, 1.5])

  # Get 3 samples, returning a 3 x 2 tensor.
  dist.sample([3])
  ```

  Arguments are broadcast when possible.

  ```python
  # Define a batch of two scalar valued Normals.
  # Both have mean 1, but different standard deviations.
  dist = tfd.Normal(loc=1., scale=[11, 22.])

  # Evaluate the pdf of both distributions on the same point, 3.0,
  # returning a length 2 tensor.
  dist.prob(3.0)
  ```

  """
  @deprecation.deprecated("2019-01-01", "The TensorFlow Distributions library has moved to " "TensorFlow Probability " "(https://github.com/tensorflow/probability). You " "should update all references to use `tfp.distributions` " "instead of `tf.distributions`.", warn_once=True)
  def __init__(self, loc, scale, validate_args=..., allow_nan_stats=..., name=...) -> None:
    """Construct Normal distributions with mean and stddev `loc` and `scale`.

    The parameters `loc` and `scale` must be shaped in a way that supports
    broadcasting (e.g. `loc + scale` is a valid operation).

    Args:
      loc: Floating point tensor; the means of the distribution(s).
      scale: Floating point tensor; the stddevs of the distribution(s).
        Must contain only positive values.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`,
        statistics (e.g., mean, mode, variance) use the value "`NaN`" to
        indicate the result is undefined. When `False`, an exception is raised
        if one or more of the statistic's batch members are undefined.
      name: Python `str` name prefixed to Ops created by this class.

    Raises:
      TypeError: if `loc` and `scale` have different `dtype`.
    """
    ...
  
  @property
  def loc(self): # -> defaultdict[Any, Any] | Any | list[Any] | object | None:
    """Distribution parameter for the mean."""
    ...
  
  @property
  def scale(self): # -> defaultdict[Any, Any] | Any | list[Any] | object | None:
    """Distribution parameter for standard deviation."""
    ...
  


class NormalWithSoftplusScale(Normal):
  """Normal with softplus applied to `scale`."""
  @deprecation.deprecated("2019-01-01", "Use `tfd.Normal(loc, tf.nn.softplus(scale)) " "instead.", warn_once=True)
  def __init__(self, loc, scale, validate_args=..., allow_nan_stats=..., name=...) -> None:
    ...
  


