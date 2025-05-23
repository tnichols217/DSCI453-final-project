"""
This type stub file was generated by pyright.
"""

import contextlib
from tensorflow.python.util.tf_export import tf_export

"""The execution context for ClusterCoordinator."""
_dispatch_context = ...
def get_current_dispatch_context(): # -> Any | None:
  ...

@contextlib.contextmanager
def with_dispatch_context(worker_obj): # -> Generator[None, Any, None]:
  ...

class DispatchContext:
  """Context entered when executing a closure on a given worker."""
  def __init__(self, worker_obj) -> None:
    ...
  
  @property
  def worker(self): # -> Any:
    ...
  
  @property
  def worker_index(self):
    ...
  
  def maybe_get_remote_value(self, ret):
    ...
  


def maybe_get_remote_value(val):
  """Gets the value of `val` if it is a `RemoteValue`."""
  ...

@tf_export("distribute.coordinator.experimental_get_current_worker_index", v1=[])
def get_current_worker_index():
  """Returns the current worker index, when called within a worker closure.

  Some parameter server training workloads may require the worker to know its
  index, for example for data sharding for reduced-variance training.

  This method may be used within a `tf.function` that is executed on a worker.
  That is, either a `dataset_fn` that runs via
  `ClusterCoordinator.create_per_worker_dataset`, or any other function
  scheduled via `ClusterCoordinator.schedule`.

  Example (sharding data by worker):

  ```python
  strategy = tf.distribute.ParameterServerStrategy(
      cluster_resolver=...)
  coordinator = (
      tf.distribute.coordinator.ClusterCoordinator(strategy))

  def dataset_fn(context):
    dataset = tf.data.Dataset.range(10)
    worker_index = (
        tf.distribute.coordinator.experimental_get_current_worker_index()
    )
    dataset = dataset.shard(
        num_shards=num_workers,
        index=worker_index,
    )
    return dataset

  @tf.function
  def per_worker_dataset_fn():
    return strategy.distribute_datasets_from_function(dataset_fn)

  per_worker_dataset = coordinator.create_per_worker_dataset(
      per_worker_dataset_fn)
  ```

  Raises:
    RuntimeError: if called from outside a `tf.function` or outside of a remote
      closure execution context (that is, on a non-worker machine).
  """
  ...

