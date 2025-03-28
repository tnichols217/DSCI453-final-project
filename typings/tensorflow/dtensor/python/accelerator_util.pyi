"""
This type stub file was generated by pyright.
"""

from typing import List, Optional
from tensorflow.python.util.tf_export import tf_export

"""Utility for working with accelerator systems."""
_INITIALIZED_ACCELERATOR_SYSTEM_TYPE = ...
def is_initialized() -> bool:
  """Returns whether accelerator system has been initialized."""
  ...

def set_initialized(value): # -> None:
  """Sets if accelerator system has been initialized."""
  ...

def initialize_multi_client_cluster(job_name: str, dtensor_jobs: List[str], client_id: int, collective_leader: str, port: Optional[int] = ..., gpu_use_nccl_communication: bool = ..., enable_coordination_service: bool = ...): # -> None:
  """Initialize GRPC servers and collectives for multi-client DTensor setup.

  This function can be used to initialize a multi-client cluster and enable
  collective ops. GRPC servers are necessary in the multi-client mode, even
  when the number of clientis is 1.

  NOTE: this function must be called in an eager context.

  Args:
    job_name: The job name used by all clients in the DTensor cluster.
    dtensor_jobs: A list of the DTensor client jobs participating in the
      cluster. Must be strings of the form "hostname:port".
    client_id: The ID of the DTensor client this function is being called in.
    collective_leader: The job/task that will be used to run collectives.
    port: The port this client's GRPC server will run on. If omitted, use the
      port from dtensor_jobs for this client.
    gpu_use_nccl_communication: if True, configure TensorFlow to use NCCL by
      default.
    enable_coordination_service: If true, enable distributed coordination
      service to make sure that workers know the devices on each other, a
      prerequisite for data transfer through cross-worker rendezvous.

  Raises:
    RuntimeError: If running inside a tf.function.
  """
  ...

@tf_export("experimental.dtensor.initialize_accelerator_system", "experimental.dtensor.initialize_tpu_system", "experimental.dtensor.initialize_multi_client", v1=[])
def initialize_accelerator_system(device_type: Optional[str] = ..., enable_coordination_service: Optional[bool] = ..., num_logical_cpu_devices: Optional[int] = ..., experimental_reset_context: Optional[bool] = ..., experimental_enable_megcore: Optional[bool] = ...) -> str:
  """Initializes accelerators and communication fabrics for DTensor.

  DTensor configures TensorFlow to run in the local mode or multi-client mode.
  - In local mode, a mesh can only use devices attached to the current process.
  - In multi-client mode, a mesh can span across devices from multiple clients.

  If `DTENSOR_JOBS` is non-empty, DTensor configures TensorFlow to run in the
  multi-client mode using the distributed runtime. In multi-client mode devices
  on different clients can communicate with each other.

  The following environment variables controls the behavior of this function.

  - `DTENSOR_JOBS`: string, a comma separated list. Each item in the list is
      of format `{hostname}:{port}`. If empty, DTensor runs in the local mode.
      Examples of valid `DTENSOR_JOBS` values:
      - 4 clients on localhost:
        `localhost:10000,localhost:10001,localhost:10002,localhost:10003`
      - 2 clients on host1, 2 clients on host2
        `host1:10000,host1:10001,host2:10000,host2:10003`
      If the hostnames are BNS addresses, the items must be sorted in
      alphabetical order.
  - `DTENSOR_CLIENT_ID`: integer, between `0` to `num_clients - 1`, to identify
      the client id of the current process. The default value is `0`.
  - `DTENSOR_JOB_NAME`: string, a string for the name of the TensorFlow job.
      The job name controls the job name section of the TensorFlow DeviceSpecs,
      e.g., `job:worker` in `/job:worker/replica:0/task:0/device:TPU:0` when
      the job name is `worker`.
      The default value is `localhost` in local mode, and
      `worker` when in the multi-client mode. All DTensor clients within the
      same multi-client cluster share the same job name.
  - `DTENSOR_USE_PARALLEL_EXECUTOR`: string, with its value being `pw` to
      specify that the backend is Pathways, and TensorFlow otherwise.

  Args:
    device_type: Type of accelerator to use, can be CPU, GPU, or TPU. If None,
      uses `tf.experimental.dtensor.preferred_device_type()`.
    enable_coordination_service: If true, enable distributed coordination
      service to make sure that workers know the devices on each other, when
      there is more than 1 client.
    num_logical_cpu_devices: the number of logical CPU devices per DTensor
      client. Default to the current number of logical CPU
      (`dtensor.num_local_devices("CPU")`),when `device_type` is CPU, otherwise
      set automatially to match the number of local GPU/TPU devices.
    experimental_reset_context: Reset the tensorflow context. Behaviors of
      existing TensorFlow objects (e.g. Tensors) are undefined. Set this to True
      as an escape hatch, if there is no clear way to refactor your code to call
      initialize_accelerator_system() before calling TensorFlow APIs that
      initialize the context.
    experimental_enable_megcore: Optionally enable megcore in backend.

  Returns:
    device_type: the type of accelerator that was initialized.
  """
  ...

@tf_export("experimental.dtensor.shutdown_accelerator_system", "experimental.dtensor.shutdown_tpu_system", v1=[])
def shutdown_accelerator_system() -> None:
  """Shuts down the accelerator system."""
  ...

