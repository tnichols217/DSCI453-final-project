"""
This type stub file was generated by pyright.
"""

import collections
from tensorflow.python.util.tf_export import tf_export

"""TensorFlow 2.x Profiler.

The profiler has two modes:
- Programmatic Mode: start(logdir), stop(), and Profiler class. Profiling starts
                     when calling start(logdir) or create a Profiler class.
                     Profiling stops when calling stop() to save to
                     TensorBoard logdir or destroying the Profiler class.
- Sampling Mode: start_server(). It will perform profiling after receiving a
                 profiling request.

NOTE: Only one active profiler session is allowed. Use of simultaneous
Programmatic Mode and Sampling Mode is undefined and will likely fail.

NOTE: The Keras TensorBoard callback will automatically perform sampled
profiling. Before enabling customized profiling, set the callback flag
"profile_batches=[]" to disable automatic sampled profiling.
"""
_profiler = ...
_profiler_lock = ...
@tf_export('profiler.experimental.ProfilerOptions', v1=[])
class ProfilerOptions(collections.namedtuple('ProfilerOptions', ['host_tracer_level', 'python_tracer_level', 'device_tracer_level', 'delay_ms'])):
  """Options for finer control over the profiler.

  Use `tf.profiler.experimental.ProfilerOptions` to control `tf.profiler`
  behavior.

  Fields:
    host_tracer_level: Adjust CPU tracing level. Values are: `1` - critical info
      only, `2` - info, `3` - verbose. [default value is `2`]
    python_tracer_level: Toggle tracing of Python function calls. Values are:
      `1` - enabled, `0` - disabled [default value is `0`]
    device_tracer_level: Adjust device (TPU/GPU) tracing level. Values are:
      `1` - enabled, `0` - disabled [default value is `1`]
    delay_ms: Requests for all hosts to start profiling at a timestamp that is
      `delay_ms` away from the current time. `delay_ms` is in milliseconds. If
      zero, each host will start profiling immediately upon receiving the
      request. Default value is `None`, allowing the profiler guess the best
      value.
  """
  def __new__(cls, host_tracer_level=..., python_tracer_level=..., device_tracer_level=..., delay_ms=...): # -> Self:
    ...
  


@tf_export('profiler.experimental.start', v1=[])
def start(logdir, options=...): # -> None:
  """Start profiling TensorFlow performance.

  Args:
    logdir: Profiling results log directory.
    options: `ProfilerOptions` namedtuple to specify miscellaneous profiler
      options. See example usage below.

  Raises:
    AlreadyExistsError: If a profiling session is already running.

  Example usage:
  ```python
  options = tf.profiler.experimental.ProfilerOptions(host_tracer_level = 3,
                                                     python_tracer_level = 1,
                                                     device_tracer_level = 1)
  tf.profiler.experimental.start('logdir_path', options = options)
  # Training code here
  tf.profiler.experimental.stop()
  ```

  To view the profiling results, launch TensorBoard and point it to `logdir`.
  Open your browser and go to `localhost:6006/#profile` to view profiling
  results.

  """
  ...

@tf_export('profiler.experimental.stop', v1=[])
def stop(save=...): # -> None:
  """Stops the current profiling session.

  The profiler session will be stopped and profile results can be saved.

  Args:
    save: An optional variable to save the results to TensorBoard. Default True.

  Raises:
    UnavailableError: If there is no active profiling session.
  """
  ...

def warmup(): # -> None:
  """Warm-up the profiler session.

  The profiler session will set up profiling context, including loading CUPTI
  library for GPU profiling. This is used for improving the accuracy of
  the profiling results.

  """
  ...

@tf_export('profiler.experimental.server.start', v1=[])
def start_server(port): # -> None:
  """Start a profiler grpc server that listens to given port.

  The profiler server will exit when the process finishes. The service is
  defined in tensorflow/core/profiler/profiler_service.proto.

  Args:
    port: port profiler server listens to.
  Example usage: ```python tf.profiler.experimental.server.start(6009) # do
    your training here.
  """
  ...

@tf_export('profiler.experimental.Profile', v1=[])
class Profile:
  """Context-manager profile API.

  Profiling will start when entering the scope, and stop and save the results to
  the logdir when exits the scope. Open TensorBoard profile tab to view results.

  Example usage:
  ```python
  with tf.profiler.experimental.Profile("/path/to/logdir"):
    # do some work
  ```
  """
  def __init__(self, logdir, options=...) -> None:
    """Creates a context manager object for profiler API.

    Args:
      logdir: profile data will save to this directory.
      options: An optional `tf.profiler.experimental.ProfilerOptions` can be
        provided to fine tune the profiler's behavior.
    """
    ...
  
  def __enter__(self): # -> None:
    ...
  
  def __exit__(self, typ, value, tb): # -> None:
    ...
  


