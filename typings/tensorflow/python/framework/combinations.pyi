"""
This type stub file was generated by pyright.
"""

from tensorflow.python.framework import test_combinations

"""This module customizes `test_combinations` for Tensorflow.

Additionally it provides `generate()`, `combine()` and `times()` with Tensorflow
customizations as a default.
"""
class EagerGraphCombination(test_combinations.TestCombination):
  """Run the test in Graph or Eager mode.

  The optional `mode` parameter controls the test's execution mode.  Its
  accepted values are "graph" or "eager" literals.
  """
  def context_managers(self, kwargs): # -> list[Any]:
    ...
  
  def parameter_modifiers(self): # -> list[OptionalParameter]:
    ...
  


class TFVersionCombination(test_combinations.TestCombination):
  """Control the execution of the test in TF1.x and TF2.

  If TF2 is enabled then a test with TF1 test is going to be skipped and vice
  versa.

  Test targets continuously run in TF2 thanks to the tensorflow.v2 TAP target.
  A test can be run in TF2 with bazel by passing --test_env=TF2_BEHAVIOR=1.
  """
  def should_execute_combination(self, kwargs): # -> tuple[Literal[False], Literal['Skipping a TF1.x test when TF2 is enabled.']] | tuple[Literal[False], Literal['Skipping a TF2 test when TF2 is not enabled.']] | tuple[Literal[True], None]:
    ...
  
  def parameter_modifiers(self): # -> list[OptionalParameter]:
    ...
  


generate = ...
combine = ...
times = ...
NamedObject = test_combinations.NamedObject
