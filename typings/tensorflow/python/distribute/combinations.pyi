"""
This type stub file was generated by pyright.
"""

import sys
from tensorflow.python.framework import test_combinations as combinations_lib
from tensorflow.python.util.tf_export import tf_export

"""This module customizes `test_combinations` for `tf.distribute.Strategy`.

Additionally it provides `generate()`, `combine()` and `times()` with
`tf.distribute.Strategy` customizations as a default.
"""
class DistributionParameter(combinations_lib.ParameterModifier):
  """Transforms arguments of type `NamedDistribution`.

  Convert all arguments of type `NamedDistribution` to the value of their
  `strategy` property.
  """
  def modified_arguments(self, kwargs, requested_parameters): # -> dict[Any, Any]:
    ...
  


class ClusterParameters(combinations_lib.ParameterModifier):
  """Adds cluster parameters if a `NamedDistribution` has it.

  It needs to be before DistributionParameter.
  """
  def modified_arguments(self, kwargs, requested_parameters): # -> dict[Any, Any]:
    ...
  


class DistributionCombination(combinations_lib.TestCombination):
  """Sets up distribution strategy for tests."""
  def should_execute_combination(self, kwargs): # -> tuple[Literal[False], LiteralString] | tuple[Literal[True], None]:
    ...
  
  def parameter_modifiers(self): # -> list[Any]:
    ...
  


class ClusterCombination(combinations_lib.TestCombination):
  """Sets up multi worker tests."""
  def parameter_modifiers(self): # -> list[ClusterParameters]:
    ...
  


class GPUCombination(combinations_lib.TestCombination):
  """Enable tests to request GPU hardware and skip non-GPU combinations.

  This class expects test_combinations to be generated with `NamedDistribution`
  wrapping instances of `tf.distribute.Strategy`.

  Optionally, the `required_gpus` argument is supported.  GPU hardware is
  required, if its value is `True` or > 0.

  Attributes:
    GPU_TEST: The environment is considered to have GPU hardware available if
              the name of the program contains "test_gpu" or "test_xla_gpu".
  """
  GPU_TEST = ...
  if sys.argv:
    GPU_TEST = ...
  def should_execute_combination(self, kwargs): # -> tuple[Literal[False], Literal['Test that doesn\'t require GPUs.']] | tuple[Literal[False], str] | tuple[Literal[True], None]:
    ...
  
  def parameter_modifiers(self): # -> list[OptionalParameter]:
    ...
  


class TPUCombination(combinations_lib.TestCombination):
  """Allow to request TPU hardware and skip non-TPU combinations.

  This class expects test_combinations to be generated with `NamedDistribution`
  wrapping instances of `tf.distribute.Strategy`.

  Optionally, the `required_tpus` parameter is supported.  TPU hardware is
  required, if its argument is `True` or > 0.

  Optionally, the `use_cloud_tpu` parameter is supported. If TPU hardware is
  required by `required_tpus`, it specifically must be a Cloud TPU (specified
  with `--tpu`) if `use_cloud_tpu` is `True`.

  Attributes:
    TPU_TEST: The environment is considered to have TPU hardware available if
              the name of the program contains "test_tpu".
  """
  TPU_TEST = ...
  if sys.argv:
    TPU_TEST = ...
  def should_execute_combination(self, kwargs): # -> tuple[Literal[False], Literal['Test that doesn\'t require TPUs.']] | tuple[Literal[False], Literal['Test requires a TPU, but it\'s not available.']] | tuple[Literal[False], Literal['Test requires a Cloud TPU, but none specified.']] | tuple[Literal[False], Literal['Test requires local TPU, but Cloud TPU specified.']] | tuple[Literal[True], None]:
    ...
  
  def parameter_modifiers(self): # -> list[OptionalParameter]:
    ...
  


class NamedDistribution:
  """Wraps a `tf.distribute.Strategy` and adds a name for test titles."""
  def __init__(self, name, distribution_fn, required_gpus=..., required_physical_gpus=..., required_tpu=..., use_cloud_tpu=..., has_chief=..., num_workers=..., num_ps=..., share_gpu=..., pool_runner_fn=..., no_xla=...) -> None:
    """Initialize NamedDistribution.

    Args:
      name: Name that will be a part of the name of the test case.
      distribution_fn: A callable that creates a `tf.distribute.Strategy`.
      required_gpus: The number of GPUs that the strategy requires. Only one of
      `required_gpus` and `required_physical_gpus` should be set.
      required_physical_gpus: Number of physical GPUs required. Only one of
      `required_gpus` and `required_physical_gpus` should be set.
      required_tpu: Whether the strategy requires TPU.
      use_cloud_tpu: Whether the strategy requires cloud TPU.
      has_chief: Whether the strategy requires a chief worker.
      num_workers: The number of workers that the strategy requires.
      num_ps: The number of parameter servers.
      share_gpu: Whether to share GPUs among workers.
      pool_runner_fn: An optional callable that returns a MultiProcessPoolRunner
        to run the test.
      no_xla: Whether to skip in XLA tests.
    """
    ...
  
  @property
  def runner(self): # -> None:
    ...
  
  @property
  def strategy(self):
    ...
  
  def __repr__(self): # -> Any:
    ...
  


tf_function = ...
no_tf_function = ...
def concat(*combined): # -> list[Any]:
  """Concats combinations."""
  ...

@tf_export("__internal__.distribute.combinations.generate", v1=[])
def generate(combinations, test_combinations=...): # -> Callable[..., type | _ParameterizedTestIter]:
  """Distributed adapter of `tf.__internal__.test.combinations.generate`.

  All tests with distributed strategy should use this one instead of
  `tf.__internal__.test.combinations.generate`. This function has support of
  strategy combinations, GPU/TPU and multi worker support.

  See `tf.__internal__.test.combinations.generate` for usage.
  """
  ...

combine = ...
times = ...
NamedObject = combinations_lib.NamedObject
_running_in_worker = ...
@tf_export("__internal__.distribute.combinations.in_main_process", v1=[])
def in_main_process(): # -> bool:
  """Whether it's in the main test process.

  This is normally used to prepare the test environment which should only happen
  in the main process.

  Returns:
    A boolean.
  """
  ...

class TestEnvironment:
  """Holds the test environment information.

  Tests should modify the attributes of the instance returned by `env()` in the
  main process if needed, and it will be passed to the worker processes each
  time a test case is run.
  """
  def __init__(self) -> None:
    ...
  
  def __setattr__(self, name, value): # -> None:
    ...
  


_env = ...
@tf_export("__internal__.distribute.combinations.env", v1=[])
def env(): # -> TestEnvironment:
  """Returns the object holds the test environment information.

  Tests should modify this in the main process if needed, and it will be passed
  to the worker processes each time a test case is run.

  Returns:
    a TestEnvironment object.
  """
  ...

_TestResult = ...
