"""
This type stub file was generated by pyright.
"""

from tensorflow.python.util import deprecation, dispatch
from tensorflow.python.util.tf_export import tf_export

"""RNN helpers for TensorFlow models."""
_concat = ...
@deprecation.deprecated(None, "Please use `keras.layers.Bidirectional(" "keras.layers.RNN(cell))`, which is equivalent to " "this API")
@tf_export(v1=["nn.bidirectional_dynamic_rnn"])
@dispatch.add_dispatch_support
def bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, sequence_length=..., initial_state_fw=..., initial_state_bw=..., dtype=..., parallel_iterations=..., swap_memory=..., time_major=..., scope=...): # -> tuple[tuple[Any | defaultdict[Any, Any] | list[Any] | object | None, Any], tuple[Any, Any]]:
  """Creates a dynamic version of bidirectional recurrent neural network.

  Takes input and builds independent forward and backward RNNs. The input_size
  of forward and backward cell must match. The initial state for both directions
  is zero by default (but can be set optionally) and no intermediate states are
  ever returned -- the network is fully unrolled for the given (passed in)
  length(s) of the sequence(s) or completely unrolled if length(s) is not
  given.

  Args:
    cell_fw: An instance of RNNCell, to be used for forward direction.
    cell_bw: An instance of RNNCell, to be used for backward direction.
    inputs: The RNN inputs.
      If time_major == False (default), this must be a tensor of shape:
        `[batch_size, max_time, ...]`, or a nested tuple of such elements.
      If time_major == True, this must be a tensor of shape: `[max_time,
        batch_size, ...]`, or a nested tuple of such elements.
    sequence_length: (optional) An int32/int64 vector, size `[batch_size]`,
      containing the actual lengths for each of the sequences in the batch. If
      not provided, all batch entries are assumed to be full sequences; and time
      reversal is applied from time `0` to `max_time` for each sequence.
    initial_state_fw: (optional) An initial state for the forward RNN. This must
      be a tensor of appropriate type and shape `[batch_size,
      cell_fw.state_size]`. If `cell_fw.state_size` is a tuple, this should be a
      tuple of tensors having shapes `[batch_size, s] for s in
      cell_fw.state_size`.
    initial_state_bw: (optional) Same as for `initial_state_fw`, but using the
      corresponding properties of `cell_bw`.
    dtype: (optional) The data type for the initial states and expected output.
      Required if initial_states are not provided or RNN states have a
      heterogeneous dtype.
    parallel_iterations: (Default: 32).  The number of iterations to run in
      parallel.  Those operations which do not have any temporal dependency and
      can be run in parallel, will be.  This parameter trades off time for
      space.  Values >> 1 use more memory but take less time, while smaller
      values use less memory but computations take longer.
    swap_memory: Transparently swap the tensors produced in forward inference
      but needed for back prop from GPU to CPU.  This allows training RNNs which
      would typically not fit on a single GPU, with very minimal (or no)
      performance penalty.
    time_major: The shape format of the `inputs` and `outputs` Tensors. If true,
      these `Tensors` must be shaped `[max_time, batch_size, depth]`. If false,
      these `Tensors` must be shaped `[batch_size, max_time, depth]`. Using
      `time_major = True` is a bit more efficient because it avoids transposes
      at the beginning and end of the RNN calculation.  However, most TensorFlow
      data is batch-major, so by default this function accepts input and emits
      output in batch-major form.
    scope: VariableScope for the created subgraph; defaults to
      "bidirectional_rnn"

  Returns:
    A tuple (outputs, output_states) where:
      outputs: A tuple (output_fw, output_bw) containing the forward and
        the backward rnn output `Tensor`.
        If time_major == False (default),
          output_fw will be a `Tensor` shaped:
          `[batch_size, max_time, cell_fw.output_size]`
          and output_bw will be a `Tensor` shaped:
          `[batch_size, max_time, cell_bw.output_size]`.
        If time_major == True,
          output_fw will be a `Tensor` shaped:
          `[max_time, batch_size, cell_fw.output_size]`
          and output_bw will be a `Tensor` shaped:
          `[max_time, batch_size, cell_bw.output_size]`.
        It returns a tuple instead of a single concatenated `Tensor`, unlike
        in the `bidirectional_rnn`. If the concatenated one is preferred,
        the forward and backward outputs can be concatenated as
        `tf.concat(outputs, 2)`.
      output_states: A tuple (output_state_fw, output_state_bw) containing
        the forward and the backward final states of bidirectional rnn.

  Raises:
    TypeError: If `cell_fw` or `cell_bw` is not an instance of `RNNCell`.
  """
  ...

@deprecation.deprecated(None, "Please use `keras.layers.RNN(cell)`, which is equivalent to this API")
@tf_export(v1=["nn.dynamic_rnn"])
@dispatch.add_dispatch_support
def dynamic_rnn(cell, inputs, sequence_length=..., initial_state=..., dtype=..., parallel_iterations=..., swap_memory=..., time_major=..., scope=...): # -> tuple[Any | defaultdict[Any, Any] | list[Any] | object | None, Any]:
  """Creates a recurrent neural network specified by RNNCell `cell`.

  Performs fully dynamic unrolling of `inputs`.

  Example:

  ```python
  # create a BasicRNNCell
  rnn_cell = tf.compat.v1.nn.rnn_cell.BasicRNNCell(hidden_size)

  # 'outputs' is a tensor of shape [batch_size, max_time, cell_state_size]

  # defining initial state
  initial_state = rnn_cell.zero_state(batch_size, dtype=tf.float32)

  # 'state' is a tensor of shape [batch_size, cell_state_size]
  outputs, state = tf.compat.v1.nn.dynamic_rnn(rnn_cell, input_data,
                                     initial_state=initial_state,
                                     dtype=tf.float32)
  ```

  ```python
  # create 2 LSTMCells
  rnn_layers = [tf.compat.v1.nn.rnn_cell.LSTMCell(size) for size in [128, 256]]

  # create a RNN cell composed sequentially of a number of RNNCells
  multi_rnn_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(rnn_layers)

  # 'outputs' is a tensor of shape [batch_size, max_time, 256]
  # 'state' is a N-tuple where N is the number of LSTMCells containing a
  # tf.nn.rnn_cell.LSTMStateTuple for each cell
  outputs, state = tf.compat.v1.nn.dynamic_rnn(cell=multi_rnn_cell,
                                     inputs=data,
                                     dtype=tf.float32)
  ```


  Args:
    cell: An instance of RNNCell.
    inputs: The RNN inputs.
      If `time_major == False` (default), this must be a `Tensor` of shape:
        `[batch_size, max_time, ...]`, or a nested tuple of such elements.
      If `time_major == True`, this must be a `Tensor` of shape: `[max_time,
        batch_size, ...]`, or a nested tuple of such elements. This may also be
        a (possibly nested) tuple of Tensors satisfying this property.  The
        first two dimensions must match across all the inputs, but otherwise the
        ranks and other shape components may differ. In this case, input to
        `cell` at each time-step will replicate the structure of these tuples,
        except for the time dimension (from which the time is taken). The input
        to `cell` at each time step will be a `Tensor` or (possibly nested)
        tuple of Tensors each with dimensions `[batch_size, ...]`.
    sequence_length: (optional) An int32/int64 vector sized `[batch_size]`. Used
      to copy-through state and zero-out outputs when past a batch element's
      sequence length.  This parameter enables users to extract the last valid
      state and properly padded outputs, so it is provided for correctness.
    initial_state: (optional) An initial state for the RNN. If `cell.state_size`
      is an integer, this must be a `Tensor` of appropriate type and shape
      `[batch_size, cell.state_size]`. If `cell.state_size` is a tuple, this
      should be a tuple of tensors having shapes `[batch_size, s] for s in
      cell.state_size`.
    dtype: (optional) The data type for the initial state and expected output.
      Required if initial_state is not provided or RNN state has a heterogeneous
      dtype.
    parallel_iterations: (Default: 32).  The number of iterations to run in
      parallel.  Those operations which do not have any temporal dependency and
      can be run in parallel, will be.  This parameter trades off time for
      space.  Values >> 1 use more memory but take less time, while smaller
      values use less memory but computations take longer.
    swap_memory: Transparently swap the tensors produced in forward inference
      but needed for back prop from GPU to CPU.  This allows training RNNs which
      would typically not fit on a single GPU, with very minimal (or no)
      performance penalty.
    time_major: The shape format of the `inputs` and `outputs` Tensors. If true,
      these `Tensors` must be shaped `[max_time, batch_size, depth]`. If false,
      these `Tensors` must be shaped `[batch_size, max_time, depth]`. Using
      `time_major = True` is a bit more efficient because it avoids transposes
      at the beginning and end of the RNN calculation.  However, most TensorFlow
      data is batch-major, so by default this function accepts input and emits
      output in batch-major form.
    scope: VariableScope for the created subgraph; defaults to "rnn".

  Returns:
    A pair (outputs, state) where:

    outputs: The RNN output `Tensor`.

      If time_major == False (default), this will be a `Tensor` shaped:
        `[batch_size, max_time, cell.output_size]`.

      If time_major == True, this will be a `Tensor` shaped:
        `[max_time, batch_size, cell.output_size]`.

      Note, if `cell.output_size` is a (possibly nested) tuple of integers
      or `TensorShape` objects, then `outputs` will be a tuple having the
      same structure as `cell.output_size`, containing Tensors having shapes
      corresponding to the shape data in `cell.output_size`.

    state: The final state.  If `cell.state_size` is an int, this
      will be shaped `[batch_size, cell.state_size]`.  If it is a
      `TensorShape`, this will be shaped `[batch_size] + cell.state_size`.
      If it is a (possibly nested) tuple of ints or `TensorShape`, this will
      be a tuple having the corresponding shapes. If cells are `LSTMCells`
      `state` will be a tuple containing a `LSTMStateTuple` for each cell.

  Raises:
    TypeError: If `cell` is not an instance of RNNCell.
    ValueError: If inputs is None or an empty list.

  @compatibility(TF2)
  `tf.compat.v1.nn.dynamic_rnn` is not compatible with eager execution and
  `tf.function`. Please use `tf.keras.layers.RNN` instead for TF2 migration.
  Take LSTM as an example, you can instantiate a `tf.keras.layers.RNN` layer
  with `tf.keras.layers.LSTMCell`, or directly via `tf.keras.layers.LSTM`. Once
  the keras layer is created, you can get the output and states by calling
  the layer with input and states. Please refer to [this
  guide](https://www.tensorflow.org/guide/keras/rnn) for more details about
  Keras RNN. You can also find more details about the difference and comparison
  between Keras RNN and TF compat v1 rnn in [this
  document](https://github.com/tensorflow/community/blob/master/rfcs/20180920-unify-rnn-interface.md)

  #### Structural Mapping to Native TF2

  Before:

  ```python
  # create 2 LSTMCells
  rnn_layers = [tf.compat.v1.nn.rnn_cell.LSTMCell(size) for size in [128, 256]]

  # create a RNN cell composed sequentially of a number of RNNCells
  multi_rnn_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(rnn_layers)

  # 'outputs' is a tensor of shape [batch_size, max_time, 256]
  # 'state' is a N-tuple where N is the number of LSTMCells containing a
  # tf.nn.rnn_cell.LSTMStateTuple for each cell
  outputs, state = tf.compat.v1.nn.dynamic_rnn(cell=multi_rnn_cell,
                                               inputs=data,
                                               dtype=tf.float32)
  ```

  After:

  ```python
  # RNN layer can take a list of cells, which will then stack them together.
  # By default, keras RNN will only return the last timestep output and will not
  # return states. If you need whole time sequence output as well as the states,
  # you can set `return_sequences` and `return_state` to True.
  rnn_layer = tf.keras.layers.RNN([tf.keras.layers.LSTMCell(128),
                                   tf.keras.layers.LSTMCell(256)],
                                  return_sequences=True,
                                  return_state=True)
  outputs, output_states = rnn_layer(inputs, states)
  ```

  #### How to Map Arguments

  | TF1 Arg Name          | TF2 Arg Name    | Note                             |
  | :-------------------- | :-------------- | :------------------------------- |
  | `cell`                | `cell`          | In the RNN layer constructor     |
  | `inputs`              | `inputs`        | In the RNN layer `__call__`      |
  | `sequence_length`     | Not used        | Adding masking layer before RNN  :
  :                       :                 : to achieve the same result.      :
  | `initial_state`       | `initial_state` | In the RNN layer `__call__`      |
  | `dtype`               | `dtype`         | In the RNN layer constructor     |
  | `parallel_iterations` | Not supported   |                                  |
  | `swap_memory`         | Not supported   |                                  |
  | `time_major`          | `time_major`    | In the RNN layer constructor     |
  | `scope`               | Not supported   |                                  |
  @end_compatibility
  """
  ...

@tf_export(v1=["nn.raw_rnn"])
@dispatch.add_dispatch_support
def raw_rnn(cell, loop_fn, parallel_iterations=..., swap_memory=..., scope=...): # -> tuple[Any, Any, Any | None]:
  """Creates an `RNN` specified by RNNCell `cell` and loop function `loop_fn`.

  **NOTE: This method is still in testing, and the API may change.**

  This function is a more primitive version of `dynamic_rnn` that provides
  more direct access to the inputs each iteration.  It also provides more
  control over when to start and finish reading the sequence, and
  what to emit for the output.

  For example, it can be used to implement the dynamic decoder of a seq2seq
  model.

  Instead of working with `Tensor` objects, most operations work with
  `TensorArray` objects directly.

  The operation of `raw_rnn`, in pseudo-code, is basically the following:

  ```python
  time = tf.constant(0, dtype=tf.int32)
  (finished, next_input, initial_state, emit_structure, loop_state) = loop_fn(
      time=time, cell_output=None, cell_state=None, loop_state=None)
  emit_ta = TensorArray(dynamic_size=True, dtype=initial_state.dtype)
  state = initial_state
  while not all(finished):
    (output, cell_state) = cell(next_input, state)
    (next_finished, next_input, next_state, emit, loop_state) = loop_fn(
        time=time + 1, cell_output=output, cell_state=cell_state,
        loop_state=loop_state)
    # Emit zeros and copy forward state for minibatch entries that are finished.
    state = tf.where(finished, state, next_state)
    emit = tf.where(finished, tf.zeros_like(emit_structure), emit)
    emit_ta = emit_ta.write(time, emit)
    # If any new minibatch entries are marked as finished, mark these.
    finished = tf.logical_or(finished, next_finished)
    time += 1
  return (emit_ta, state, loop_state)
  ```

  with the additional properties that output and state may be (possibly nested)
  tuples, as determined by `cell.output_size` and `cell.state_size`, and
  as a result the final `state` and `emit_ta` may themselves be tuples.

  A simple implementation of `dynamic_rnn` via `raw_rnn` looks like this:

  ```python
  inputs = tf.compat.v1.placeholder(shape=(max_time, batch_size, input_depth),
                          dtype=tf.float32)
  sequence_length = tf.compat.v1.placeholder(shape=(batch_size,),
  dtype=tf.int32)
  inputs_ta = tf.TensorArray(dtype=tf.float32, size=max_time)
  inputs_ta = inputs_ta.unstack(inputs)

  cell = tf.compat.v1.nn.rnn_cell.LSTMCell(num_units)

  def loop_fn(time, cell_output, cell_state, loop_state):
    emit_output = cell_output  # == None for time == 0
    if cell_output is None:  # time == 0
      next_cell_state = cell.zero_state(batch_size, tf.float32)
    else:
      next_cell_state = cell_state
    elements_finished = (time >= sequence_length)
    finished = tf.reduce_all(elements_finished)
    next_input = tf.cond(
        finished,
        lambda: tf.zeros([batch_size, input_depth], dtype=tf.float32),
        lambda: inputs_ta.read(time))
    next_loop_state = None
    return (elements_finished, next_input, next_cell_state,
            emit_output, next_loop_state)

  outputs_ta, final_state, _ = raw_rnn(cell, loop_fn)
  outputs = outputs_ta.stack()
  ```

  Args:
    cell: An instance of RNNCell.
    loop_fn: A callable that takes inputs `(time, cell_output, cell_state,
      loop_state)` and returns the tuple `(finished, next_input,
      next_cell_state, emit_output, next_loop_state)`. Here `time` is an int32
      scalar `Tensor`, `cell_output` is a `Tensor` or (possibly nested) tuple of
      tensors as determined by `cell.output_size`, and `cell_state` is a
      `Tensor` or (possibly nested) tuple of tensors, as determined by the
      `loop_fn` on its first call (and should match `cell.state_size`).
      The outputs are: `finished`, a boolean `Tensor` of
      shape `[batch_size]`, `next_input`: the next input to feed to `cell`,
      `next_cell_state`: the next state to feed to `cell`,
      and `emit_output`: the output to store for this iteration.  Note that
        `emit_output` should be a `Tensor` or (possibly nested) tuple of tensors
        which is aggregated in the `emit_ta` inside the `while_loop`. For the
        first call to `loop_fn`, the `emit_output` corresponds to the
        `emit_structure` which is then used to determine the size of the
        `zero_tensor` for the `emit_ta` (defaults to `cell.output_size`). For
        the subsequent calls to the `loop_fn`, the `emit_output` corresponds to
        the actual output tensor that is to be aggregated in the `emit_ta`. The
        parameter `cell_state` and output `next_cell_state` may be either a
        single or (possibly nested) tuple of tensors.  The parameter
        `loop_state` and output `next_loop_state` may be either a single or
        (possibly nested) tuple of `Tensor` and `TensorArray` objects.  This
        last parameter may be ignored by `loop_fn` and the return value may be
        `None`.  If it is not `None`, then the `loop_state` will be propagated
        through the RNN loop, for use purely by `loop_fn` to keep track of its
        own state. The `next_loop_state` parameter returned may be `None`.  The
        first call to `loop_fn` will be `time = 0`, `cell_output = None`,
      `cell_state = None`, and `loop_state = None`.  For this call: The
        `next_cell_state` value should be the value with which to initialize the
        cell's state.  It may be a final state from a previous RNN or it may be
        the output of `cell.zero_state()`.  It should be a (possibly nested)
        tuple structure of tensors. If `cell.state_size` is an integer, this
        must be a `Tensor` of appropriate type and shape `[batch_size,
        cell.state_size]`. If `cell.state_size` is a `TensorShape`, this must be
        a `Tensor` of appropriate type and shape `[batch_size] +
        cell.state_size`. If `cell.state_size` is a (possibly nested) tuple of
        ints or `TensorShape`, this will be a tuple having the corresponding
        shapes. The `emit_output` value may be either `None` or a (possibly
        nested) tuple structure of tensors, e.g., `(tf.zeros(shape_0,
        dtype=dtype_0), tf.zeros(shape_1, dtype=dtype_1))`. If this first
        `emit_output` return value is `None`, then the `emit_ta` result of
        `raw_rnn` will have the same structure and dtypes as `cell.output_size`.
        Otherwise `emit_ta` will have the same structure, shapes (prepended with
        a `batch_size` dimension), and dtypes as `emit_output`.  The actual
        values returned for `emit_output` at this initializing call are ignored.
        Note, this emit structure must be consistent across all time steps.
    parallel_iterations: (Default: 32).  The number of iterations to run in
      parallel.  Those operations which do not have any temporal dependency and
      can be run in parallel, will be.  This parameter trades off time for
      space.  Values >> 1 use more memory but take less time, while smaller
      values use less memory but computations take longer.
    swap_memory: Transparently swap the tensors produced in forward inference
      but needed for back prop from GPU to CPU.  This allows training RNNs which
      would typically not fit on a single GPU, with very minimal (or no)
      performance penalty.
    scope: VariableScope for the created subgraph; defaults to "rnn".

  Returns:
    A tuple `(emit_ta, final_state, final_loop_state)` where:

    `emit_ta`: The RNN output `TensorArray`.
       If `loop_fn` returns a (possibly nested) set of Tensors for
       `emit_output` during initialization, (inputs `time = 0`,
       `cell_output = None`, and `loop_state = None`), then `emit_ta` will
       have the same structure, dtypes, and shapes as `emit_output` instead.
       If `loop_fn` returns `emit_output = None` during this call,
       the structure of `cell.output_size` is used:
       If `cell.output_size` is a (possibly nested) tuple of integers
       or `TensorShape` objects, then `emit_ta` will be a tuple having the
       same structure as `cell.output_size`, containing TensorArrays whose
       elements' shapes correspond to the shape data in `cell.output_size`.

    `final_state`: The final cell state.  If `cell.state_size` is an int, this
      will be shaped `[batch_size, cell.state_size]`.  If it is a
      `TensorShape`, this will be shaped `[batch_size] + cell.state_size`.
      If it is a (possibly nested) tuple of ints or `TensorShape`, this will
      be a tuple having the corresponding shapes.

    `final_loop_state`: The final loop state as returned by `loop_fn`.

  Raises:
    TypeError: If `cell` is not an instance of RNNCell, or `loop_fn` is not
      a `callable`.
  """
  ...

@deprecation.deprecated(None, "Please use `keras.layers.RNN(cell, unroll=True)`, " "which is equivalent to this API")
@tf_export(v1=["nn.static_rnn"])
@dispatch.add_dispatch_support
def static_rnn(cell, inputs, initial_state=..., dtype=..., sequence_length=..., scope=...): # -> tuple[list[Any], Any | defaultdict[Any, Any] | list[Any] | None]:
  """Creates a recurrent neural network specified by RNNCell `cell`.

  The simplest form of RNN network generated is:

  ```python
    state = cell.zero_state(...)
    outputs = []
    for input_ in inputs:
      output, state = cell(input_, state)
      outputs.append(output)
    return (outputs, state)
  ```
  However, a few other options are available:

  An initial state can be provided.
  If the sequence_length vector is provided, dynamic calculation is performed.
  This method of calculation does not compute the RNN steps past the maximum
  sequence length of the minibatch (thus saving computational time),
  and properly propagates the state at an example's sequence length
  to the final state output.

  The dynamic calculation performed is, at time `t` for batch row `b`,

  ```python
    (output, state)(b, t) =
      (t >= sequence_length(b))
        ? (zeros(cell.output_size), states(b, sequence_length(b) - 1))
        : cell(input(b, t), state(b, t - 1))
  ```

  Args:
    cell: An instance of RNNCell.
    inputs: A length T list of inputs, each a `Tensor` of shape `[batch_size,
      input_size]`, or a nested tuple of such elements.
    initial_state: (optional) An initial state for the RNN. If `cell.state_size`
      is an integer, this must be a `Tensor` of appropriate type and shape
      `[batch_size, cell.state_size]`. If `cell.state_size` is a tuple, this
      should be a tuple of tensors having shapes `[batch_size, s] for s in
      cell.state_size`.
    dtype: (optional) The data type for the initial state and expected output.
      Required if initial_state is not provided or RNN state has a heterogeneous
      dtype.
    sequence_length: Specifies the length of each sequence in inputs. An int32
      or int64 vector (tensor) size `[batch_size]`, values in `[0, T)`.
    scope: VariableScope for the created subgraph; defaults to "rnn".

  Returns:
    A pair (outputs, state) where:

    - outputs is a length T list of outputs (one for each input), or a nested
      tuple of such elements.
    - state is the final state

  Raises:
    TypeError: If `cell` is not an instance of RNNCell.
    ValueError: If `inputs` is `None` or an empty list, or if the input depth
      (column size) cannot be inferred from inputs via shape inference.
  """
  ...

@deprecation.deprecated(None, "Please use `keras.layers.RNN(cell, stateful=True)`, " "which is equivalent to this API")
@tf_export(v1=["nn.static_state_saving_rnn"])
@dispatch.add_dispatch_support
def static_state_saving_rnn(cell, inputs, state_saver, state_name, sequence_length=..., scope=...): # -> tuple[list[Any], Any | defaultdict[Any, Any] | list[Any] | object | None]:
  """RNN that accepts a state saver for time-truncated RNN calculation.

  Args:
    cell: An instance of `RNNCell`.
    inputs: A length T list of inputs, each a `Tensor` of shape `[batch_size,
      input_size]`.
    state_saver: A state saver object with methods `state` and `save_state`.
    state_name: Python string or tuple of strings.  The name to use with the
      state_saver. If the cell returns tuples of states (i.e., `cell.state_size`
      is a tuple) then `state_name` should be a tuple of strings having the same
      length as `cell.state_size`.  Otherwise it should be a single string.
    sequence_length: (optional) An int32/int64 vector size [batch_size]. See the
      documentation for rnn() for more details about sequence_length.
    scope: VariableScope for the created subgraph; defaults to "rnn".

  Returns:
    A pair (outputs, state) where:
      outputs is a length T list of outputs (one for each input)
      states is the final state

  Raises:
    TypeError: If `cell` is not an instance of RNNCell.
    ValueError: If `inputs` is `None` or an empty list, or if the arity and
     type of `state_name` does not match that of `cell.state_size`.
  """
  ...

@deprecation.deprecated(None, "Please use `keras.layers.Bidirectional(" "keras.layers.RNN(cell, unroll=True))`, which is " "equivalent to this API")
@tf_export(v1=["nn.static_bidirectional_rnn"])
@dispatch.add_dispatch_support
def static_bidirectional_rnn(cell_fw, cell_bw, inputs, initial_state_fw=..., initial_state_bw=..., dtype=..., sequence_length=..., scope=...): # -> tuple[Any | defaultdict[Any, Any] | list[Any] | None, Any | defaultdict[Any, Any] | list[Any] | None, Any | defaultdict[Any, Any] | list[Any] | None]:
  """Creates a bidirectional recurrent neural network.

  Similar to the unidirectional case above (rnn) but takes input and builds
  independent forward and backward RNNs with the final forward and backward
  outputs depth-concatenated, such that the output will have the format
  [time][batch][cell_fw.output_size + cell_bw.output_size]. The input_size of
  forward and backward cell must match. The initial state for both directions
  is zero by default (but can be set optionally) and no intermediate states are
  ever returned -- the network is fully unrolled for the given (passed in)
  length(s) of the sequence(s) or completely unrolled if length(s) is not given.

  Args:
    cell_fw: An instance of RNNCell, to be used for forward direction.
    cell_bw: An instance of RNNCell, to be used for backward direction.
    inputs: A length T list of inputs, each a tensor of shape [batch_size,
      input_size], or a nested tuple of such elements.
    initial_state_fw: (optional) An initial state for the forward RNN. This must
      be a tensor of appropriate type and shape `[batch_size,
      cell_fw.state_size]`. If `cell_fw.state_size` is a tuple, this should be a
      tuple of tensors having shapes `[batch_size, s] for s in
      cell_fw.state_size`.
    initial_state_bw: (optional) Same as for `initial_state_fw`, but using the
      corresponding properties of `cell_bw`.
    dtype: (optional) The data type for the initial state.  Required if either
      of the initial states are not provided.
    sequence_length: (optional) An int32/int64 vector, size `[batch_size]`,
      containing the actual lengths for each of the sequences.
    scope: VariableScope for the created subgraph; defaults to
      "bidirectional_rnn"

  Returns:
    A tuple (outputs, output_state_fw, output_state_bw) where:
      outputs is a length `T` list of outputs (one for each input), which
        are depth-concatenated forward and backward outputs.
      output_state_fw is the final state of the forward rnn.
      output_state_bw is the final state of the backward rnn.

  Raises:
    TypeError: If `cell_fw` or `cell_bw` is not an instance of `RNNCell`.
    ValueError: If inputs is None or an empty list.
  """
  ...

