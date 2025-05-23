"""
This type stub file was generated by pyright.
"""

"""Writer class for `DebugEvent` protos in tfdbg v2."""
DEFAULT_CIRCULAR_BUFFER_SIZE = ...
class DebugEventsWriter:
  """A writer for TF debugging events. Used by tfdbg v2."""
  def __init__(self, dump_root, tfdbg_run_id, circular_buffer_size=...) -> None:
    """Construct a DebugEventsWriter object.

    NOTE: Given the same `dump_root`, all objects from this constructor
      will point to the same underlying set of writers. In other words, they
      will write to the same set of debug events files in the `dump_root`
      folder.

    Args:
      dump_root: The root directory for dumping debug data. If `dump_root` does
        not exist as a directory, it will be created.
      tfdbg_run_id: Debugger Run ID.
      circular_buffer_size: Size of the circular buffer for each of the two
        execution-related debug events files: with the following suffixes: -
          .execution - .graph_execution_traces If <= 0, the circular-buffer
          behavior will be abolished in the constructed object.
    """
    ...
  
  def WriteSourceFile(self, source_file): # -> None:
    """Write a SourceFile proto with the writer.

    Args:
      source_file: A SourceFile proto, describing the content of a source file
        involved in the execution of the debugged TensorFlow program.
    """
    ...
  
  def WriteStackFrameWithId(self, stack_frame_with_id): # -> None:
    """Write a StackFrameWithId proto with the writer.

    Args:
      stack_frame_with_id: A StackFrameWithId proto, describing the content a
        stack frame involved in the execution of the debugged TensorFlow
        program.
    """
    ...
  
  def WriteGraphOpCreation(self, graph_op_creation): # -> None:
    """Write a GraphOpCreation proto with the writer.

    Args:
      graph_op_creation: A GraphOpCreation proto, describing the details of the
        creation of an op inside a TensorFlow Graph.
    """
    ...
  
  def WriteDebuggedGraph(self, debugged_graph): # -> None:
    """Write a DebuggedGraph proto with the writer.

    Args:
      debugged_graph: A DebuggedGraph proto, describing the details of a
        TensorFlow Graph that has completed its construction.
    """
    ...
  
  def WriteExecution(self, execution): # -> None:
    """Write a Execution proto with the writer.

    Args:
      execution: An Execution proto, describing a TensorFlow op or graph
        execution event.
    """
    ...
  
  def WriteGraphExecutionTrace(self, graph_execution_trace): # -> None:
    """Write a GraphExecutionTrace proto with the writer.

    Args:
      graph_execution_trace: A GraphExecutionTrace proto, concerning the value
        of an intermediate tensor or a list of intermediate tensors that are
        computed during the graph's execution.
    """
    ...
  
  def RegisterDeviceAndGetId(self, device_name): # -> int:
    ...
  
  def FlushNonExecutionFiles(self): # -> None:
    """Flush the non-execution debug event files."""
    ...
  
  def FlushExecutionFiles(self): # -> None:
    """Flush the execution debug event files.

    Causes the current content of the cyclic buffers to be written to
    the .execution and .graph_execution_traces debug events files.
    Also clears those cyclic buffers.
    """
    ...
  
  def Close(self): # -> None:
    """Close the writer."""
    ...
  
  @property
  def dump_root(self): # -> Any:
    ...
  


