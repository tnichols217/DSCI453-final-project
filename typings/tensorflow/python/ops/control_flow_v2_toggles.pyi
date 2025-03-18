"""
This type stub file was generated by pyright.
"""

from tensorflow.python.util.tf_export import tf_export

"""API for enabling v2 control flow."""
@tf_export(v1=["enable_control_flow_v2"])
def enable_control_flow_v2(): # -> None:
  """Use control flow v2.

  control flow v2 (cfv2) is an improved version of control flow in TensorFlow
  with support for higher order derivatives. Enabling cfv2 will change the
  graph/function representation of control flow, e.g., `tf.while_loop` and
  `tf.cond` will generate functional `While` and `If` ops instead of low-level
  `Switch`, `Merge` etc. ops. Note: Importing and running graphs exported
  with old control flow will still be supported.

  Calling tf.enable_control_flow_v2() lets you opt-in to this TensorFlow 2.0
  feature.

  Note: v2 control flow is always enabled inside of tf.function. Calling this
  function is not required.
  """
  ...

@tf_export(v1=["disable_control_flow_v2"])
def disable_control_flow_v2(): # -> None:
  """Opts out of control flow v2.

  Note: v2 control flow is always enabled inside of tf.function. Calling this
  function has no effect in that case.

  If your code needs tf.disable_control_flow_v2() to be called to work
  properly please file a bug.
  """
  ...

@tf_export(v1=["control_flow_v2_enabled"])
def control_flow_v2_enabled(): # -> bool:
  """Returns `True` if v2 control flow is enabled.

  Note: v2 control flow is always enabled inside of tf.function.
  """
  ...

