"""
This type stub file was generated by pyright.
"""

from tensorflow.python.util.tf_export import tf_export

"""Provides a utility class for preemption detection and recovery."""
_preemption_watcher_initialization_counter = ...
_preemption_handling_counter = ...
_PREEMPTION_KEY = ...
@tf_export("distribute.experimental.PreemptionWatcher", v1=[])
class PreemptionWatcher:
  """Watch preemption signal and store it.

  Notice: Currently only support Borg TPU environment with TPUClusterResolver.

  This class provides a way to monitor the preemption signal during training on
  TPU. It will start a background thread to watch the training process, trying
  to fetch preemption message from the coordination service. When preemption
  happens, the preempted worker will write the preemption message to the
  coordination service. Thus getting a non-empty preemption message means there
  is a preemption happened.

  User can use the preemption message as a reliable preemption indicator, and
  then set the coordinator to reconnect to the TPU worker instead of a fully
  restart triggered by Borg. For example, a training process with
  preemption recovery will be like:

  ```python
  keep_running = True
  preemption_watcher = None
  while keep_running:
    try:
      # Initialize TPU cluster and stratygy.
      resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
      tf.config.experimental_connect_to_cluster(resolver)
      tf.tpu.experimental.initialize_tpu_system(resolver)
      strategy = tf.distribute.TPUStrategy(resolver)

      # PreemptionWatcher must be created after connected to cluster.
      preemption_watcher = tf.distribute.experimental.PreemptionWatcher()
      train_model(strategy)
      keep_running = False
    except Exception as e:
      if preemption_watcher and preemption_watcher.preemption_message:
        preemption_watcher.block_until_worker_exit()
        keep_running = True
      else:
        raise e
  ```

  Attributes:
    preemption_message: A variable to store the preemption message fetched from
      the coordination service. If it is not None, then there is a preemption
      happened.
    platform: A PlatformDevice to indicate the current job's platform. Refer to
      failure_handling_util.py for the definition of enum class PlatformDevice.
  """
  def __init__(self) -> None:
    ...
  
  @property
  def preemption_message(self): # -> None:
    """Returns the preemption message."""
    ...
  
  def block_until_worker_exit(self): # -> None:
    """Block coordinator until workers exit.

    In some rare cases, another error could be raised during the
    preemption grace period. This will cause the coordinator to reconnect to the
    same TPU workers, which will be killed later. It prevents the coordinator to
    reconnect to new TPU workers, and falls back to a hard restart. To avoid
    this situation, this method will block the coordinator to reconnect until
    workers exit. This method will be a no-op for non-TPU platform.
    """
    ...
  


