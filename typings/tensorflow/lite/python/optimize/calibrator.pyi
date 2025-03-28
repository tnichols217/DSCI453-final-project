"""
This type stub file was generated by pyright.
"""

from tensorflow.lite.python.convert_phase import Component, SubComponent, convert_phase

"""Python wrapper for post training quantization with calibration."""
_calibration_wrapper = ...
def add_intermediate_tensors(model_content): # -> Any:
  """Adds intermediate tensors to fused op if needed."""
  ...

class Calibrator:
  """Calibrates a floating point model and then quantizes it.

  This is an internal class, not a public interface.
  """
  def __init__(self, model_content, custom_op_registerers_by_name=..., custom_op_registerers_by_func=...) -> None:
    """Constructor.

    Args:
      model_content: Content of a TF-Lite Flatbuffer file.
      custom_op_registerers_by_name: List of str (symbol names) that take a
        pointer to a MutableOpResolver and register custom ops.
      custom_op_registerers_by_func: List of functions that take a pointer to a
        MutableOpResolver and register custom ops.

    Raises:
      ValueError: If the calibrator was unable to open the model.
    """
    ...
  
  @convert_phase(Component.OPTIMIZE_TFLITE_MODEL, SubComponent.QUANTIZE_USING_DEPRECATED_QUANTIZER)
  def calibrate_and_quantize(self, dataset_gen, input_type, output_type, allow_float, activations_type=..., bias_type=..., resize_input=..., disable_per_channel=...): # -> Any:
    """Calibrates the model with specified generator and then quantizes it.

    The input shapes of the calibrator are resized with the calibration data if
    `resize_input` is set.

    Returns:
      A quantized model.

    Args:
      dataset_gen: A generator that generates calibration samples.
      input_type: A tf.dtype representing the desired real-value input type.
      output_type: A tf.dtype representing the desired real-value output type.
      allow_float: A boolean. False if the resulting model cannot perform float
        computation, useful when targeting an integer-only backend. If False, an
        error will be thrown if an operation cannot be quantized, otherwise the
        model will fallback to float ops.
      activations_type: A tf.dtype representing the desired type for
        activations.
      bias_type: A tf.dtype representing the desired type for bias.
      resize_input: A boolean. True if the shape of the sample data is different
        from the input.
      disable_per_channel: A boolean. True if disabling per-channel
        quantization.
    """
    ...
  
  @convert_phase(Component.OPTIMIZE_TFLITE_MODEL, SubComponent.QUANTIZE_USING_DEPRECATED_QUANTIZER)
  def calibrate_and_quantize_single(self, dataset_gen, input_type, output_type, allow_float, op_output_name, resize_input=...): # -> Any:
    """Calibrates the model with specified generator and then quantizes it.

    Only the single op with output op_output_name will be quantized.
    The input shapes of the calibrator are resized with the calibration data.

    Returns:
      A quantized model.

    Args:
      dataset_gen: A generator that generates calibration samples.
      input_type: A tf.dtype representing the desired real-value input type.
      output_type: A tf.dtype representing the desired real-value output type.
      allow_float: A boolean. False if the resulting model cannot perform float
        computation, useful when targeting an integer-only backend. If False, an
        error will be thrown if an operation cannot be quantized, otherwise the
        model will fallback to float ops.
      op_output_name: A string, only this op will be quantized.
      resize_input: A boolean. True if the shape of the sample data is different
        from the input.
    """
    ...
  
  @convert_phase(Component.OPTIMIZE_TFLITE_MODEL, SubComponent.CALIBRATE)
  def calibrate(self, dataset_gen): # -> Any:
    """Calibrates the model with specified generator.

    Returns:
      A model with min and max calibration stats.

    Args:
      dataset_gen: A generator that generates calibration samples.
    """
    ...
  


