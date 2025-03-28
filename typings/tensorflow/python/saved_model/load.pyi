"""
This type stub file was generated by pyright.
"""

from tensorflow.python.eager import function
from tensorflow.python.util.tf_export import tf_export

"""Import a trackable object from a SavedModel."""
_LOAD_V2_LABEL = ...
_BUILT_IN_REGISTRATIONS = ...
class _WrapperFunction(function.ConcreteFunction):
  """A class wraps a concrete function to handle different distributed contexts.

  The reason for wrapping a concrete function is because the _captured_inputs
  fields used for in-replica context and cross-replica context are different.
  When `load()` is called from within a tf.distribute.strategy scope, the
  captured inputs are distributed variables. When using these distributed
  variables during calling the function, we need different approaches when it is
  in-replica and when it is not in-replica. When it is in replica, naturally we
  should use the corresponding component of the distributed variable; when it is
  not in-replica, calling the function should mean that it is constructing a
  graph that is not actually going to be used. A typical use case is when
  constructing a functional model. In this case, return a placeholder with a
  control dependency to ensure that is never accessed.
  """
  def __init__(self, concrete_function) -> None:
    ...
  


class Loader:
  """Helper class to load an object-based SavedModel."""
  def __init__(self, object_graph_proto, saved_model_proto, export_dir, ckpt_options, save_options, filters) -> None:
    ...
  
  def adjust_debug_info_func_names(self, debug_info):
    """Rewrite func names in the debug info by using the concrete func names."""
    ...
  
  def get(self, node_id):
    ...
  


@tf_export("saved_model.load", v1=["saved_model.load_v2"])
def load(export_dir, tags=..., options=...): # -> AutoTrackable | None:
  """Load a SavedModel from `export_dir`.

  Signatures associated with the SavedModel are available as functions:

  ```python
  imported = tf.saved_model.load(path)
  f = imported.signatures["serving_default"]
  print(f(x=tf.constant([[1.]])))
  ```

  Objects exported with `tf.saved_model.save` additionally have trackable
  objects and functions assigned to attributes:

  ```python
  exported = tf.train.Checkpoint(v=tf.Variable(3.))
  exported.f = tf.function(
      lambda x: exported.v * x,
      input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
  tf.saved_model.save(exported, path)
  imported = tf.saved_model.load(path)
  assert 3. == imported.v.numpy()
  assert 6. == imported.f(x=tf.constant(2.)).numpy()
  ```

  _Loading Keras models_

  Keras models are trackable, so they can be saved to SavedModel. The object
  returned by `tf.saved_model.load` is not a Keras object (i.e. doesn't have
  `.fit`, `.predict`, etc. methods). A few attributes and functions are still
  available: `.variables`, `.trainable_variables` and `.__call__`.

  ```python
  model = tf.keras.Model(...)
  tf.saved_model.save(model, path)
  imported = tf.saved_model.load(path)
  outputs = imported(inputs)
  ```

  Use `tf.keras.models.load_model` to restore the Keras model.

  _Importing SavedModels from TensorFlow 1.x_

  1.x SavedModels APIs have a flat graph instead of `tf.function` objects.
  These SavedModels will be loaded with the following attributes:

  * `.signatures`: A dictionary mapping signature names to functions.
  * `.prune(feeds, fetches) `: A method which allows you to extract
    functions for new subgraphs. This is equivalent to importing the SavedModel
    and naming feeds and fetches in a Session from TensorFlow 1.x.

    ```python
    imported = tf.saved_model.load(path_to_v1_saved_model)
    pruned = imported.prune("x:0", "out:0")
    pruned(tf.ones([]))
    ```

    See `tf.compat.v1.wrap_function` for details.
  * `.variables`: A list of imported variables.
  * `.graph`: The whole imported graph.
  * `.restore(save_path)`: A function that restores variables from a checkpoint
    saved from `tf.compat.v1.Saver`.

  _Consuming SavedModels asynchronously_

  When consuming SavedModels asynchronously (the producer is a separate
  process), the SavedModel directory will appear before all files have been
  written, and `tf.saved_model.load` will fail if pointed at an incomplete
  SavedModel. Rather than checking for the directory, check for
  "saved_model_dir/saved_model.pb". This file is written atomically as the last
  `tf.saved_model.save` file operation.

  Args:
    export_dir: The SavedModel directory to load from.
    tags: A tag or sequence of tags identifying the MetaGraph to load. Optional
      if the SavedModel contains a single MetaGraph, as for those exported from
      `tf.saved_model.save`.
    options: `tf.saved_model.LoadOptions` object that specifies options for
      loading.

  Returns:
    A trackable object with a `signatures` attribute mapping from signature
    keys to functions. If the SavedModel was exported by `tf.saved_model.save`,
    it also points to trackable objects, functions, debug info which it has been
    saved.

  Raises:
    ValueError: If `tags` don't match a MetaGraph in the SavedModel.
  """
  ...

@tf_export("__internal__.saved_model.load_partial", v1=[])
def load_partial(export_dir, filters, tags=..., options=...): # -> dict[Any, Any] | dict[str, Any | AutoTrackable | None]:
  """Partially load a SavedModel (saved from V2).

  Similar to `tf.saved_model.load`, but with an additional argument that
  lets you specify which nodes to load.
  `tf.saved_model.load_partial(export_dir, ["root"])` and
  `tf.saved_model.load(export_dir)` are equivalent.

  Note: This only works for SavedModels saved with TensorFlow V2 from
  `tf.saved_model.save` or Keras.

  In Tensorflow V2, SavedModel stores the **object graph** of the saved object.
  The graph contains nodes (`tf.Module`, `tf.Variable`, `tf.function`, Keras
  layers, etc.) and edges that are the name of the attributes connecting the
  objects.

  *Example 1*

  ```
  model = tf.Module()
  model.child_layer = tf.Module()
  model.child_layer.v = tf.Variable(5.)
  tf.saved_model.save(model, '/tmp/model')
  loaded = tf.__internal__.saved_model.load_partial(
  ...   '/tmp/model',
  ...   ['root.child_layer', 'root.child_layer.v'])
  loaded['root.child_layer'].v.numpy()
  5.
  loaded['root.child_layer'].v is loaded['root.child_layer.v']
  True

  *Example 2*
  model = tf.Module()
  model.child_layer = tf.Module()
  model.child_layer.v = tf.Variable(5.)
  >>>
  tf.saved_model.save(model, '/tmp/model')
  # Create a variable
  new_variable = tf.Variable(0.)
  loaded = tf.__internal__.saved_model.load_partial(
  ...   '/tmp/model',
  ...   {'root.child_layer': None, 'root.child_layer.v': new_variable})
  loaded['root.child_layer'].v.numpy()
  5.
  new_variable.numpy()
  5.
  ```

  **Loading under different distribution strategies**
  You can load different parts of the model under different distribution
  strategies. Note that this is very experimental so use with care.

  ```
  model = tf.Module()
  model.layer_1 = tf.Module()
  model.layer_1.v = tf.Variable(5.)
  model.layer_2 = tf.Module()
  model.layer_2.v = tf.Variable(7.)
  tf.saved_model.save(model, '/tmp/model')
  # Load with no strategy
  loaded = tf.__internal__.saved_model.load_partial(
  ...   '/tmp/model',
  ...   ['root.layer_1'])
  loaded['root.layer_1'].v
  <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=5.0>
  strategy = tf.distribute.MirroredStrategy()
  with strategy.scope():
  ...   loaded2 = tf.__internal__.saved_model.load_partial(
  ...     '/tmp/model',
  ...     ['root.layer_2'])
  loaded2['root.layer_2'].v
  MirroredVariable:{
      0: <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=7.0>
  }
  ```

  Args:
    export_dir: The SavedModel directory to load from.
    filters: A list or dictionary where each element or key is a string
      path to nodes that should be loaded. Node paths consist of all the child
      attribute names to reach that node in the form: `root.{attribute_name}`.
      The loader will load all of the specified nodes and their recursive
      descendants. When this option is defined, the loader will return a
      dictionary mapping the node paths to the loaded objects.
    tags: A tag or sequence of tags identifying the MetaGraph to load. Optional
      if the SavedModel contains a single MetaGraph, as for those exported from
      `tf.saved_model.save`.
    options: `tf.saved_model.LoadOptions` object that specifies options for
      loading.

  Returns:
    A dictionary mapping node paths from the filter to loaded objects.
  """
  ...

def is_tf2_saved_model(export_dir): # -> bool:
  """Identifies if an exported SavedModel is a TF2 SavedModel.

  There are differences in SavedModel semantics between TF1 and TF2 that are
  documented here:
  https://www.tensorflow.org/guide/migrate/saved_model#savedmodel. This helper
  util function serves to distinguish the TF1 vs TF2 semantics used when
  exporting SavedModels.

  Args:
    export_dir: The SavedModel directory to load from.

  Returns:
    True if TF2 SavedModel semantics are used, False if TF1 SavedModel semantics
    are used.
  """
  ...

