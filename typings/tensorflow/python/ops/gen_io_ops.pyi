"""
This type stub file was generated by pyright.
"""

from tensorflow.security.fuzzing.py import annotation_types as _atypes
from tensorflow.python.util.deprecation import deprecated_endpoints
from tensorflow.python.util import dispatch as _dispatch
from tensorflow.python.util.tf_export import tf_export
from typing import Any, TypeVar
from typing_extensions import Annotated

"""Python wrappers around TensorFlow ops.

This file is MACHINE GENERATED! Do not edit.
"""
def fixed_length_record_reader(record_bytes: int, header_bytes: int = ..., footer_bytes: int = ..., hop_bytes: int = ..., container: str = ..., shared_name: str = ..., name=...) -> Annotated[Any, _atypes.String]:
  r"""A Reader that outputs fixed-length records from a file.

  Args:
    record_bytes: An `int`. Number of bytes in the record.
    header_bytes: An optional `int`. Defaults to `0`.
      Number of bytes in the header, defaults to 0.
    footer_bytes: An optional `int`. Defaults to `0`.
      Number of bytes in the footer, defaults to 0.
    hop_bytes: An optional `int`. Defaults to `0`.
      Number of bytes to hop before each read. Default of 0 means using
      record_bytes.
    container: An optional `string`. Defaults to `""`.
      If non-empty, this reader is placed in the given container.
      Otherwise, a default container is used.
    shared_name: An optional `string`. Defaults to `""`.
      If non-empty, this reader is named in the given bucket
      with this shared_name. Otherwise, the node name is used instead.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type mutable `string`.
  """
  ...

FixedLengthRecordReader = ...
def fixed_length_record_reader_eager_fallback(record_bytes: int, header_bytes: int, footer_bytes: int, hop_bytes: int, container: str, shared_name: str, name, ctx) -> Annotated[Any, _atypes.String]:
  ...

def fixed_length_record_reader_v2(record_bytes: int, header_bytes: int = ..., footer_bytes: int = ..., hop_bytes: int = ..., container: str = ..., shared_name: str = ..., encoding: str = ..., name=...) -> Annotated[Any, _atypes.Resource]:
  r"""A Reader that outputs fixed-length records from a file.

  Args:
    record_bytes: An `int`. Number of bytes in the record.
    header_bytes: An optional `int`. Defaults to `0`.
      Number of bytes in the header, defaults to 0.
    footer_bytes: An optional `int`. Defaults to `0`.
      Number of bytes in the footer, defaults to 0.
    hop_bytes: An optional `int`. Defaults to `0`.
      Number of bytes to hop before each read. Default of 0 means using
      record_bytes.
    container: An optional `string`. Defaults to `""`.
      If non-empty, this reader is placed in the given container.
      Otherwise, a default container is used.
    shared_name: An optional `string`. Defaults to `""`.
      If non-empty, this reader is named in the given bucket
      with this shared_name. Otherwise, the node name is used instead.
    encoding: An optional `string`. Defaults to `""`.
      The type of encoding for the file. Currently ZLIB and GZIP
      are supported. Defaults to none.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `resource`.
  """
  ...

FixedLengthRecordReaderV2 = ...
def fixed_length_record_reader_v2_eager_fallback(record_bytes: int, header_bytes: int, footer_bytes: int, hop_bytes: int, container: str, shared_name: str, encoding: str, name, ctx) -> Annotated[Any, _atypes.Resource]:
  ...

def identity_reader(container: str = ..., shared_name: str = ..., name=...) -> Annotated[Any, _atypes.String]:
  r"""A Reader that outputs the queued work as both the key and value.

  To use, enqueue strings in a Queue.  ReaderRead will take the front
  work string and output (work, work).

  Args:
    container: An optional `string`. Defaults to `""`.
      If non-empty, this reader is placed in the given container.
      Otherwise, a default container is used.
    shared_name: An optional `string`. Defaults to `""`.
      If non-empty, this reader is named in the given bucket
      with this shared_name. Otherwise, the node name is used instead.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type mutable `string`.
  """
  ...

IdentityReader = ...
def identity_reader_eager_fallback(container: str, shared_name: str, name, ctx) -> Annotated[Any, _atypes.String]:
  ...

def identity_reader_v2(container: str = ..., shared_name: str = ..., name=...) -> Annotated[Any, _atypes.Resource]:
  r"""A Reader that outputs the queued work as both the key and value.

  To use, enqueue strings in a Queue.  ReaderRead will take the front
  work string and output (work, work).

  Args:
    container: An optional `string`. Defaults to `""`.
      If non-empty, this reader is placed in the given container.
      Otherwise, a default container is used.
    shared_name: An optional `string`. Defaults to `""`.
      If non-empty, this reader is named in the given bucket
      with this shared_name. Otherwise, the node name is used instead.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `resource`.
  """
  ...

IdentityReaderV2 = ...
def identity_reader_v2_eager_fallback(container: str, shared_name: str, name, ctx) -> Annotated[Any, _atypes.Resource]:
  ...

def lmdb_reader(container: str = ..., shared_name: str = ..., name=...) -> Annotated[Any, _atypes.String]:
  r"""A Reader that outputs the records from a LMDB file.

  Args:
    container: An optional `string`. Defaults to `""`.
      If non-empty, this reader is placed in the given container.
      Otherwise, a default container is used.
    shared_name: An optional `string`. Defaults to `""`.
      If non-empty, this reader is named in the given bucket
      with this shared_name. Otherwise, the node name is used instead.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type mutable `string`.
  """
  ...

LMDBReader = ...
def lmdb_reader_eager_fallback(container: str, shared_name: str, name, ctx) -> Annotated[Any, _atypes.String]:
  ...

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('io.matching_files', v1=['io.matching_files', 'matching_files'])
@deprecated_endpoints('matching_files')
def matching_files(pattern: Annotated[Any, _atypes.String], name=...) -> Annotated[Any, _atypes.String]:
  r"""Returns the set of files matching one or more glob patterns.

  Note that this routine only supports wildcard characters in the
  basename portion of the pattern, not in the directory portion.
  Note also that the order of filenames returned is deterministic.

  Args:
    pattern: A `Tensor` of type `string`.
      Shell wildcard pattern(s). Scalar or vector of type string.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  ...

MatchingFiles = ...
_dispatcher_for_matching_files = matching_files._tf_type_based_dispatcher.Dispatch
def matching_files_eager_fallback(pattern: Annotated[Any, _atypes.String], name, ctx) -> Annotated[Any, _atypes.String]:
  ...

def merge_v2_checkpoints(checkpoint_prefixes: Annotated[Any, _atypes.String], destination_prefix: Annotated[Any, _atypes.String], delete_old_dirs: bool = ..., allow_missing_files: bool = ..., name=...): # -> object | Operation | None:
  r"""V2 format specific: merges the metadata files of sharded checkpoints.  The

  result is one logical checkpoint, with one physical metadata file and renamed
  data files.

  Intended for "grouping" multiple checkpoints in a sharded checkpoint setup.

  If delete_old_dirs is true, attempts to delete recursively the dirname of each
  path in the input checkpoint_prefixes.  This is useful when those paths are non
  user-facing temporary locations.

  If allow_missing_files is true, merges the checkpoint prefixes as long as
  at least one file exists. Otherwise, if no files exist, an error will be thrown.
  The default value for allow_missing_files is false.

  Args:
    checkpoint_prefixes: A `Tensor` of type `string`.
      prefixes of V2 checkpoints to merge.
    destination_prefix: A `Tensor` of type `string`.
      scalar.  The desired final prefix.  Allowed to be the same
      as one of the checkpoint_prefixes.
    delete_old_dirs: An optional `bool`. Defaults to `True`. see above.
    allow_missing_files: An optional `bool`. Defaults to `False`. see above.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  ...

MergeV2Checkpoints = ...
def merge_v2_checkpoints_eager_fallback(checkpoint_prefixes: Annotated[Any, _atypes.String], destination_prefix: Annotated[Any, _atypes.String], delete_old_dirs: bool, allow_missing_files: bool, name, ctx): # -> None:
  ...

def read_file(filename: Annotated[Any, _atypes.String], name=...) -> Annotated[Any, _atypes.String]:
  r"""Reads and outputs the entire contents of the input filename.

  Args:
    filename: A `Tensor` of type `string`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  ...

ReadFile = ...
def read_file_eager_fallback(filename: Annotated[Any, _atypes.String], name, ctx) -> Annotated[Any, _atypes.String]:
  ...

def reader_num_records_produced(reader_handle: Annotated[Any, _atypes.String], name=...) -> Annotated[Any, _atypes.Int64]:
  r"""Returns the number of records this Reader has produced.

  This is the same as the number of ReaderRead executions that have
  succeeded.

  Args:
    reader_handle: A `Tensor` of type mutable `string`. Handle to a Reader.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int64`.
  """
  ...

ReaderNumRecordsProduced = ...
def reader_num_records_produced_eager_fallback(reader_handle: Annotated[Any, _atypes.String], name, ctx) -> Annotated[Any, _atypes.Int64]:
  ...

def reader_num_records_produced_v2(reader_handle: Annotated[Any, _atypes.Resource], name=...) -> Annotated[Any, _atypes.Int64]:
  r"""Returns the number of records this Reader has produced.

  This is the same as the number of ReaderRead executions that have
  succeeded.

  Args:
    reader_handle: A `Tensor` of type `resource`. Handle to a Reader.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int64`.
  """
  ...

ReaderNumRecordsProducedV2 = ...
def reader_num_records_produced_v2_eager_fallback(reader_handle: Annotated[Any, _atypes.Resource], name, ctx) -> Annotated[Any, _atypes.Int64]:
  ...

def reader_num_work_units_completed(reader_handle: Annotated[Any, _atypes.String], name=...) -> Annotated[Any, _atypes.Int64]:
  r"""Returns the number of work units this Reader has finished processing.

  Args:
    reader_handle: A `Tensor` of type mutable `string`. Handle to a Reader.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int64`.
  """
  ...

ReaderNumWorkUnitsCompleted = ...
def reader_num_work_units_completed_eager_fallback(reader_handle: Annotated[Any, _atypes.String], name, ctx) -> Annotated[Any, _atypes.Int64]:
  ...

def reader_num_work_units_completed_v2(reader_handle: Annotated[Any, _atypes.Resource], name=...) -> Annotated[Any, _atypes.Int64]:
  r"""Returns the number of work units this Reader has finished processing.

  Args:
    reader_handle: A `Tensor` of type `resource`. Handle to a Reader.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int64`.
  """
  ...

ReaderNumWorkUnitsCompletedV2 = ...
def reader_num_work_units_completed_v2_eager_fallback(reader_handle: Annotated[Any, _atypes.Resource], name, ctx) -> Annotated[Any, _atypes.Int64]:
  ...

_ReaderReadOutput = ...
def reader_read(reader_handle: Annotated[Any, _atypes.String], queue_handle: Annotated[Any, _atypes.String], name=...): # -> ReaderRead:
  r"""Returns the next record (key, value pair) produced by a Reader.

  Will dequeue from the input queue if necessary (e.g. when the
  Reader needs to start reading from a new file since it has finished
  with the previous file).

  Args:
    reader_handle: A `Tensor` of type mutable `string`. Handle to a Reader.
    queue_handle: A `Tensor` of type mutable `string`.
      Handle to a Queue, with string work items.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (key, value).

    key: A `Tensor` of type `string`.
    value: A `Tensor` of type `string`.
  """
  ...

ReaderRead = ...
def reader_read_eager_fallback(reader_handle: Annotated[Any, _atypes.String], queue_handle: Annotated[Any, _atypes.String], name, ctx):
  ...

_ReaderReadUpToOutput = ...
def reader_read_up_to(reader_handle: Annotated[Any, _atypes.String], queue_handle: Annotated[Any, _atypes.String], num_records: Annotated[Any, _atypes.Int64], name=...): # -> ReaderReadUpTo:
  r"""Returns up to `num_records` (key, value) pairs produced by a Reader.

  Will dequeue from the input queue if necessary (e.g. when the
  Reader needs to start reading from a new file since it has finished
  with the previous file).
  It may return less than `num_records` even before the last batch.

  Args:
    reader_handle: A `Tensor` of type mutable `string`. Handle to a `Reader`.
    queue_handle: A `Tensor` of type mutable `string`.
      Handle to a `Queue`, with string work items.
    num_records: A `Tensor` of type `int64`.
      number of records to read from `Reader`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (keys, values).

    keys: A `Tensor` of type `string`.
    values: A `Tensor` of type `string`.
  """
  ...

ReaderReadUpTo = ...
def reader_read_up_to_eager_fallback(reader_handle: Annotated[Any, _atypes.String], queue_handle: Annotated[Any, _atypes.String], num_records: Annotated[Any, _atypes.Int64], name, ctx):
  ...

_ReaderReadUpToV2Output = ...
def reader_read_up_to_v2(reader_handle: Annotated[Any, _atypes.Resource], queue_handle: Annotated[Any, _atypes.Resource], num_records: Annotated[Any, _atypes.Int64], name=...): # -> ReaderReadUpToV2:
  r"""Returns up to `num_records` (key, value) pairs produced by a Reader.

  Will dequeue from the input queue if necessary (e.g. when the
  Reader needs to start reading from a new file since it has finished
  with the previous file).
  It may return less than `num_records` even before the last batch.

  Args:
    reader_handle: A `Tensor` of type `resource`. Handle to a `Reader`.
    queue_handle: A `Tensor` of type `resource`.
      Handle to a `Queue`, with string work items.
    num_records: A `Tensor` of type `int64`.
      number of records to read from `Reader`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (keys, values).

    keys: A `Tensor` of type `string`.
    values: A `Tensor` of type `string`.
  """
  ...

ReaderReadUpToV2 = ...
def reader_read_up_to_v2_eager_fallback(reader_handle: Annotated[Any, _atypes.Resource], queue_handle: Annotated[Any, _atypes.Resource], num_records: Annotated[Any, _atypes.Int64], name, ctx): # -> ReaderReadUpToV2:
  ...

_ReaderReadV2Output = ...
def reader_read_v2(reader_handle: Annotated[Any, _atypes.Resource], queue_handle: Annotated[Any, _atypes.Resource], name=...): # -> ReaderReadV2:
  r"""Returns the next record (key, value pair) produced by a Reader.

  Will dequeue from the input queue if necessary (e.g. when the
  Reader needs to start reading from a new file since it has finished
  with the previous file).

  Args:
    reader_handle: A `Tensor` of type `resource`. Handle to a Reader.
    queue_handle: A `Tensor` of type `resource`.
      Handle to a Queue, with string work items.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (key, value).

    key: A `Tensor` of type `string`.
    value: A `Tensor` of type `string`.
  """
  ...

ReaderReadV2 = ...
def reader_read_v2_eager_fallback(reader_handle: Annotated[Any, _atypes.Resource], queue_handle: Annotated[Any, _atypes.Resource], name, ctx): # -> ReaderReadV2:
  ...

def reader_reset(reader_handle: Annotated[Any, _atypes.String], name=...): # -> Operation:
  r"""Restore a Reader to its initial clean state.

  Args:
    reader_handle: A `Tensor` of type mutable `string`. Handle to a Reader.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  ...

ReaderReset = ...
def reader_reset_eager_fallback(reader_handle: Annotated[Any, _atypes.String], name, ctx):
  ...

def reader_reset_v2(reader_handle: Annotated[Any, _atypes.Resource], name=...): # -> object | Operation | None:
  r"""Restore a Reader to its initial clean state.

  Args:
    reader_handle: A `Tensor` of type `resource`. Handle to a Reader.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  ...

ReaderResetV2 = ...
def reader_reset_v2_eager_fallback(reader_handle: Annotated[Any, _atypes.Resource], name, ctx): # -> None:
  ...

def reader_restore_state(reader_handle: Annotated[Any, _atypes.String], state: Annotated[Any, _atypes.String], name=...): # -> Operation:
  r"""Restore a reader to a previously saved state.

  Not all Readers support being restored, so this can produce an
  Unimplemented error.

  Args:
    reader_handle: A `Tensor` of type mutable `string`. Handle to a Reader.
    state: A `Tensor` of type `string`.
      Result of a ReaderSerializeState of a Reader with type
      matching reader_handle.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  ...

ReaderRestoreState = ...
def reader_restore_state_eager_fallback(reader_handle: Annotated[Any, _atypes.String], state: Annotated[Any, _atypes.String], name, ctx):
  ...

def reader_restore_state_v2(reader_handle: Annotated[Any, _atypes.Resource], state: Annotated[Any, _atypes.String], name=...): # -> object | Operation | None:
  r"""Restore a reader to a previously saved state.

  Not all Readers support being restored, so this can produce an
  Unimplemented error.

  Args:
    reader_handle: A `Tensor` of type `resource`. Handle to a Reader.
    state: A `Tensor` of type `string`.
      Result of a ReaderSerializeState of a Reader with type
      matching reader_handle.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  ...

ReaderRestoreStateV2 = ...
def reader_restore_state_v2_eager_fallback(reader_handle: Annotated[Any, _atypes.Resource], state: Annotated[Any, _atypes.String], name, ctx): # -> None:
  ...

def reader_serialize_state(reader_handle: Annotated[Any, _atypes.String], name=...) -> Annotated[Any, _atypes.String]:
  r"""Produce a string tensor that encodes the state of a Reader.

  Not all Readers support being serialized, so this can produce an
  Unimplemented error.

  Args:
    reader_handle: A `Tensor` of type mutable `string`. Handle to a Reader.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  ...

ReaderSerializeState = ...
def reader_serialize_state_eager_fallback(reader_handle: Annotated[Any, _atypes.String], name, ctx) -> Annotated[Any, _atypes.String]:
  ...

def reader_serialize_state_v2(reader_handle: Annotated[Any, _atypes.Resource], name=...) -> Annotated[Any, _atypes.String]:
  r"""Produce a string tensor that encodes the state of a Reader.

  Not all Readers support being serialized, so this can produce an
  Unimplemented error.

  Args:
    reader_handle: A `Tensor` of type `resource`. Handle to a Reader.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  ...

ReaderSerializeStateV2 = ...
def reader_serialize_state_v2_eager_fallback(reader_handle: Annotated[Any, _atypes.Resource], name, ctx) -> Annotated[Any, _atypes.String]:
  ...

TV_Restore_dt = TypeVar("TV_Restore_dt", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
def restore(file_pattern: Annotated[Any, _atypes.String], tensor_name: Annotated[Any, _atypes.String], dt: TV_Restore_dt, preferred_shard: int = ..., name=...) -> Annotated[Any, TV_Restore_dt]:
  r"""Restores a tensor from checkpoint files.

  Reads a tensor stored in one or several files. If there are several files (for
  instance because a tensor was saved as slices), `file_pattern` may contain
  wildcard symbols (`*` and `?`) in the filename portion only, not in the
  directory portion.

  If a `file_pattern` matches several files, `preferred_shard` can be used to hint
  in which file the requested tensor is likely to be found. This op will first
  open the file at index `preferred_shard` in the list of matching files and try
  to restore tensors from that file.  Only if some tensors or tensor slices are
  not found in that first file, then the Op opens all the files. Setting
  `preferred_shard` to match the value passed as the `shard` input
  of a matching `Save` Op may speed up Restore.  This attribute only affects
  performance, not correctness.  The default value -1 means files are processed in
  order.

  See also `RestoreSlice`.

  Args:
    file_pattern: A `Tensor` of type `string`.
      Must have a single element. The pattern of the files from
      which we read the tensor.
    tensor_name: A `Tensor` of type `string`.
      Must have a single element. The name of the tensor to be
      restored.
    dt: A `tf.DType`. The type of the tensor to be restored.
    preferred_shard: An optional `int`. Defaults to `-1`.
      Index of file to open first if multiple files match
      `file_pattern`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dt`.
  """
  ...

Restore = ...
def restore_eager_fallback(file_pattern: Annotated[Any, _atypes.String], tensor_name: Annotated[Any, _atypes.String], dt: TV_Restore_dt, preferred_shard: int, name, ctx) -> Annotated[Any, TV_Restore_dt]:
  ...

TV_RestoreSlice_dt = TypeVar("TV_RestoreSlice_dt", _atypes.BFloat16, _atypes.Bool, _atypes.Complex128, _atypes.Complex64, _atypes.Float16, _atypes.Float32, _atypes.Float64, _atypes.Float8e4m3fn, _atypes.Float8e5m2, _atypes.Half, _atypes.Int16, _atypes.Int32, _atypes.Int4, _atypes.Int64, _atypes.Int8, _atypes.QInt16, _atypes.QInt32, _atypes.QInt8, _atypes.QUInt16, _atypes.QUInt8, _atypes.Resource, _atypes.String, _atypes.UInt16, _atypes.UInt32, _atypes.UInt4, _atypes.UInt64, _atypes.UInt8, _atypes.Variant)
def restore_slice(file_pattern: Annotated[Any, _atypes.String], tensor_name: Annotated[Any, _atypes.String], shape_and_slice: Annotated[Any, _atypes.String], dt: TV_RestoreSlice_dt, preferred_shard: int = ..., name=...) -> Annotated[Any, TV_RestoreSlice_dt]:
  r"""Restores a tensor from checkpoint files.

  This is like `Restore` except that restored tensor can be listed as filling
  only a slice of a larger tensor.  `shape_and_slice` specifies the shape of the
  larger tensor and the slice that the restored tensor covers.

  The `shape_and_slice` input has the same format as the
  elements of the `shapes_and_slices` input of the `SaveSlices` op.

  Args:
    file_pattern: A `Tensor` of type `string`.
      Must have a single element. The pattern of the files from
      which we read the tensor.
    tensor_name: A `Tensor` of type `string`.
      Must have a single element. The name of the tensor to be
      restored.
    shape_and_slice: A `Tensor` of type `string`.
      Scalar. The shapes and slice specifications to use when
      restoring a tensors.
    dt: A `tf.DType`. The type of the tensor to be restored.
    preferred_shard: An optional `int`. Defaults to `-1`.
      Index of file to open first if multiple files match
      `file_pattern`. See the documentation for `Restore`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dt`.
  """
  ...

RestoreSlice = ...
def restore_slice_eager_fallback(file_pattern: Annotated[Any, _atypes.String], tensor_name: Annotated[Any, _atypes.String], shape_and_slice: Annotated[Any, _atypes.String], dt: TV_RestoreSlice_dt, preferred_shard: int, name, ctx) -> Annotated[Any, TV_RestoreSlice_dt]:
  ...

def restore_v2(prefix: Annotated[Any, _atypes.String], tensor_names: Annotated[Any, _atypes.String], shape_and_slices: Annotated[Any, _atypes.String], dtypes, name=...): # -> object | Operation | tuple[Any, ...] | list[Any]:
  r"""Restores tensors from a V2 checkpoint.

  For backward compatibility with the V1 format, this Op currently allows
  restoring from a V1 checkpoint as well:
    - This Op first attempts to find the V2 index file pointed to by "prefix", and
      if found proceed to read it as a V2 checkpoint;
    - Otherwise the V1 read path is invoked.
  Relying on this behavior is not recommended, as the ability to fall back to read
  V1 might be deprecated and eventually removed.

  By default, restores the named tensors in full.  If the caller wishes to restore
  specific slices of stored tensors, "shape_and_slices" should be non-empty
  strings and correspondingly well-formed.

  Callers must ensure all the named tensors are indeed stored in the checkpoint.

  Args:
    prefix: A `Tensor` of type `string`.
      Must have a single element.  The prefix of a V2 checkpoint.
    tensor_names: A `Tensor` of type `string`.
      shape {N}.  The names of the tensors to be restored.
    shape_and_slices: A `Tensor` of type `string`.
      shape {N}.  The slice specs of the tensors to be restored.
      Empty strings indicate that they are non-partitioned tensors.
    dtypes: A list of `tf.DTypes` that has length `>= 1`.
      shape {N}.  The list of expected dtype for the tensors.  Must match
      those stored in the checkpoint.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `dtypes`.
  """
  ...

RestoreV2 = ...
def restore_v2_eager_fallback(prefix: Annotated[Any, _atypes.String], tensor_names: Annotated[Any, _atypes.String], shape_and_slices: Annotated[Any, _atypes.String], dtypes, name, ctx): # -> object:
  ...

def save(filename: Annotated[Any, _atypes.String], tensor_names: Annotated[Any, _atypes.String], data, name=...): # -> object | Operation | None:
  r"""Saves the input tensors to disk.

  The size of `tensor_names` must match the number of tensors in `data`. `data[i]`
  is written to `filename` with name `tensor_names[i]`.

  See also `SaveSlices`.

  Args:
    filename: A `Tensor` of type `string`.
      Must have a single element. The name of the file to which we write
      the tensor.
    tensor_names: A `Tensor` of type `string`.
      Shape `[N]`. The names of the tensors to be saved.
    data: A list of `Tensor` objects. `N` tensors to save.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  ...

Save = ...
def save_eager_fallback(filename: Annotated[Any, _atypes.String], tensor_names: Annotated[Any, _atypes.String], data, name, ctx): # -> None:
  ...

def save_slices(filename: Annotated[Any, _atypes.String], tensor_names: Annotated[Any, _atypes.String], shapes_and_slices: Annotated[Any, _atypes.String], data, name=...): # -> object | Operation | None:
  r"""Saves input tensors slices to disk.

  This is like `Save` except that tensors can be listed in the saved file as being
  a slice of a larger tensor.  `shapes_and_slices` specifies the shape of the
  larger tensor and the slice that this tensor covers. `shapes_and_slices` must
  have as many elements as `tensor_names`.

  Elements of the `shapes_and_slices` input must either be:

  *  The empty string, in which case the corresponding tensor is
     saved normally.
  *  A string of the form `dim0 dim1 ... dimN-1 slice-spec` where the
     `dimI` are the dimensions of the larger tensor and `slice-spec`
     specifies what part is covered by the tensor to save.

  `slice-spec` itself is a `:`-separated list: `slice0:slice1:...:sliceN-1`
  where each `sliceI` is either:

  *  The string `-` meaning that the slice covers all indices of this dimension
  *  `start,length` where `start` and `length` are integers.  In that
     case the slice covers `length` indices starting at `start`.

  See also `Save`.

  Args:
    filename: A `Tensor` of type `string`.
      Must have a single element. The name of the file to which we write the
      tensor.
    tensor_names: A `Tensor` of type `string`.
      Shape `[N]`. The names of the tensors to be saved.
    shapes_and_slices: A `Tensor` of type `string`.
      Shape `[N]`.  The shapes and slice specifications to use when
      saving the tensors.
    data: A list of `Tensor` objects. `N` tensors to save.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  ...

SaveSlices = ...
def save_slices_eager_fallback(filename: Annotated[Any, _atypes.String], tensor_names: Annotated[Any, _atypes.String], shapes_and_slices: Annotated[Any, _atypes.String], data, name, ctx): # -> None:
  ...

def save_v2(prefix: Annotated[Any, _atypes.String], tensor_names: Annotated[Any, _atypes.String], shape_and_slices: Annotated[Any, _atypes.String], tensors, name=...): # -> object | Operation | None:
  r"""Saves tensors in V2 checkpoint format.

  By default, saves the named tensors in full.  If the caller wishes to save
  specific slices of full tensors, "shape_and_slices" should be non-empty strings
  and correspondingly well-formed.

  Args:
    prefix: A `Tensor` of type `string`.
      Must have a single element. The prefix of the V2 checkpoint to which we
      write the tensors.
    tensor_names: A `Tensor` of type `string`.
      shape {N}. The names of the tensors to be saved.
    shape_and_slices: A `Tensor` of type `string`.
      shape {N}.  The slice specs of the tensors to be saved.
      Empty strings indicate that they are non-partitioned tensors.
    tensors: A list of `Tensor` objects. `N` tensors to save.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  ...

SaveV2 = ...
def save_v2_eager_fallback(prefix: Annotated[Any, _atypes.String], tensor_names: Annotated[Any, _atypes.String], shape_and_slices: Annotated[Any, _atypes.String], tensors, name, ctx): # -> None:
  ...

def sharded_filename(basename: Annotated[Any, _atypes.String], shard: Annotated[Any, _atypes.Int32], num_shards: Annotated[Any, _atypes.Int32], name=...) -> Annotated[Any, _atypes.String]:
  r"""Generate a sharded filename. The filename is printf formatted as

     %s-%05d-of-%05d, basename, shard, num_shards.

  Args:
    basename: A `Tensor` of type `string`.
    shard: A `Tensor` of type `int32`.
    num_shards: A `Tensor` of type `int32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  ...

ShardedFilename = ...
def sharded_filename_eager_fallback(basename: Annotated[Any, _atypes.String], shard: Annotated[Any, _atypes.Int32], num_shards: Annotated[Any, _atypes.Int32], name, ctx) -> Annotated[Any, _atypes.String]:
  ...

def sharded_filespec(basename: Annotated[Any, _atypes.String], num_shards: Annotated[Any, _atypes.Int32], name=...) -> Annotated[Any, _atypes.String]:
  r"""Generate a glob pattern matching all sharded file names.

  Args:
    basename: A `Tensor` of type `string`.
    num_shards: A `Tensor` of type `int32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  ...

ShardedFilespec = ...
def sharded_filespec_eager_fallback(basename: Annotated[Any, _atypes.String], num_shards: Annotated[Any, _atypes.Int32], name, ctx) -> Annotated[Any, _atypes.String]:
  ...

def tf_record_reader(container: str = ..., shared_name: str = ..., compression_type: str = ..., name=...) -> Annotated[Any, _atypes.String]:
  r"""A Reader that outputs the records from a TensorFlow Records file.

  Args:
    container: An optional `string`. Defaults to `""`.
      If non-empty, this reader is placed in the given container.
      Otherwise, a default container is used.
    shared_name: An optional `string`. Defaults to `""`.
      If non-empty, this reader is named in the given bucket
      with this shared_name. Otherwise, the node name is used instead.
    compression_type: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type mutable `string`.
  """
  ...

TFRecordReader = ...
def tf_record_reader_eager_fallback(container: str, shared_name: str, compression_type: str, name, ctx) -> Annotated[Any, _atypes.String]:
  ...

def tf_record_reader_v2(container: str = ..., shared_name: str = ..., compression_type: str = ..., name=...) -> Annotated[Any, _atypes.Resource]:
  r"""A Reader that outputs the records from a TensorFlow Records file.

  Args:
    container: An optional `string`. Defaults to `""`.
      If non-empty, this reader is placed in the given container.
      Otherwise, a default container is used.
    shared_name: An optional `string`. Defaults to `""`.
      If non-empty, this reader is named in the given bucket
      with this shared_name. Otherwise, the node name is used instead.
    compression_type: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `resource`.
  """
  ...

TFRecordReaderV2 = ...
def tf_record_reader_v2_eager_fallback(container: str, shared_name: str, compression_type: str, name, ctx) -> Annotated[Any, _atypes.Resource]:
  ...

def text_line_reader(skip_header_lines: int = ..., container: str = ..., shared_name: str = ..., name=...) -> Annotated[Any, _atypes.String]:
  r"""A Reader that outputs the lines of a file delimited by '\n'.

  Args:
    skip_header_lines: An optional `int`. Defaults to `0`.
      Number of lines to skip from the beginning of every file.
    container: An optional `string`. Defaults to `""`.
      If non-empty, this reader is placed in the given container.
      Otherwise, a default container is used.
    shared_name: An optional `string`. Defaults to `""`.
      If non-empty, this reader is named in the given bucket
      with this shared_name. Otherwise, the node name is used instead.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type mutable `string`.
  """
  ...

TextLineReader = ...
def text_line_reader_eager_fallback(skip_header_lines: int, container: str, shared_name: str, name, ctx) -> Annotated[Any, _atypes.String]:
  ...

def text_line_reader_v2(skip_header_lines: int = ..., container: str = ..., shared_name: str = ..., name=...) -> Annotated[Any, _atypes.Resource]:
  r"""A Reader that outputs the lines of a file delimited by '\n'.

  Args:
    skip_header_lines: An optional `int`. Defaults to `0`.
      Number of lines to skip from the beginning of every file.
    container: An optional `string`. Defaults to `""`.
      If non-empty, this reader is placed in the given container.
      Otherwise, a default container is used.
    shared_name: An optional `string`. Defaults to `""`.
      If non-empty, this reader is named in the given bucket
      with this shared_name. Otherwise, the node name is used instead.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `resource`.
  """
  ...

TextLineReaderV2 = ...
def text_line_reader_v2_eager_fallback(skip_header_lines: int, container: str, shared_name: str, name, ctx) -> Annotated[Any, _atypes.Resource]:
  ...

def whole_file_reader(container: str = ..., shared_name: str = ..., name=...) -> Annotated[Any, _atypes.String]:
  r"""A Reader that outputs the entire contents of a file as a value.

  To use, enqueue filenames in a Queue.  The output of ReaderRead will
  be a filename (key) and the contents of that file (value).

  Args:
    container: An optional `string`. Defaults to `""`.
      If non-empty, this reader is placed in the given container.
      Otherwise, a default container is used.
    shared_name: An optional `string`. Defaults to `""`.
      If non-empty, this reader is named in the given bucket
      with this shared_name. Otherwise, the node name is used instead.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type mutable `string`.
  """
  ...

WholeFileReader = ...
def whole_file_reader_eager_fallback(container: str, shared_name: str, name, ctx) -> Annotated[Any, _atypes.String]:
  ...

def whole_file_reader_v2(container: str = ..., shared_name: str = ..., name=...) -> Annotated[Any, _atypes.Resource]:
  r"""A Reader that outputs the entire contents of a file as a value.

  To use, enqueue filenames in a Queue.  The output of ReaderRead will
  be a filename (key) and the contents of that file (value).

  Args:
    container: An optional `string`. Defaults to `""`.
      If non-empty, this reader is placed in the given container.
      Otherwise, a default container is used.
    shared_name: An optional `string`. Defaults to `""`.
      If non-empty, this reader is named in the given bucket
      with this shared_name. Otherwise, the node name is used instead.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `resource`.
  """
  ...

WholeFileReaderV2 = ...
def whole_file_reader_v2_eager_fallback(container: str, shared_name: str, name, ctx) -> Annotated[Any, _atypes.Resource]:
  ...

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('io.write_file', v1=['io.write_file', 'write_file'])
@deprecated_endpoints('write_file')
def write_file(filename: Annotated[Any, _atypes.String], contents: Annotated[Any, _atypes.String], name=...): # -> object | _dispatcher_for_write_file | Operation | None:
  r"""Writes `contents` to the file at input `filename`.

  Creates the file and recursively creates directory if it does not exist.

  Args:
    filename: A `Tensor` of type `string`.
      scalar. The name of the file to which we write the contents.
    contents: A `Tensor` of type `string`.
      scalar. The content to be written to the output file.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  ...

WriteFile = ...
_dispatcher_for_write_file = write_file._tf_type_based_dispatcher.Dispatch
def write_file_eager_fallback(filename: Annotated[Any, _atypes.String], contents: Annotated[Any, _atypes.String], name, ctx): # -> None:
  ...

