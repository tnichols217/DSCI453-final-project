"""
This type stub file was generated by pyright.
"""

from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.tf_export import tf_export

"""SavedModel builder implementation."""
_SAVE_BUILDER_LABEL = ...
@tf_export("__internal__.saved_model.SavedModelBuilder", v1=[])
class _SavedModelBuilder:
  """Builds the `SavedModel` protocol buffer and saves variables and assets.

  The `SavedModelBuilder` class provides the functionality to build a
  `SavedModel` protocol buffer. Specifically, this allows multiple meta
  graphs to be saved as part of a single language-neutral `SavedModel`,
  while sharing variables and assets.

  To build a SavedModel, the first meta graph must be saved with variables.
  Subsequent meta graphs will simply be saved with their graph definitions. If
  assets need to be saved and written or copied to disk, they can be provided
  when the meta graph def is added. If multiple meta graph defs are associated
  an asset of the same name, only the first version is retained.

  Each meta graph added to the SavedModel must be annotated with tags. The tags
  provide a means to identify the specific meta graph to load and restore, along
  with the shared set of variables and assets.

  Typical usage for the `SavedModelBuilder`:

  ```python
  ...
  builder = tf.compat.v1.saved_model.Builder(export_dir)

  with tf.compat.v1.Session(graph=tf.Graph()) as sess:
    ...
    builder.add_meta_graph_and_variables(sess,
                                    ["foo-tag"],
                                    signature_def_map=foo_signatures,
                                    assets_list=foo_assets)
  ...

  with tf.compat.v1.Session(graph=tf.Graph()) as sess:
    ...
    builder.add_meta_graph(["bar-tag", "baz-tag"])
  ...

  builder.save()
  ```

  Note: This function will only be available through the v1 compatibility
  library as tf.compat.v1.saved_model.builder.SavedModelBuilder or
  tf.compat.v1.saved_model.Builder. Tensorflow 2.0 will introduce a new
  object-based method of creating SavedModels.
  """
  def __init__(self, export_dir) -> None:
    ...
  
  def add_meta_graph(self, tags, signature_def_map=..., assets_list=..., clear_devices=..., init_op=..., train_op=..., saver=...): # -> None:
    """Adds the current meta graph to the SavedModel.

    Creates a Saver in the current scope and uses the Saver to export the meta
    graph def. Invoking this API requires the `add_meta_graph_and_variables()`
    API to have been invoked before.

    Args:
      tags: The set of tags to annotate the meta graph def with.
      signature_def_map: The map of signature defs to be added to the meta graph
        def.
      assets_list: Assets to be saved with SavedModel. Note
          that this list should be a subset of the assets saved as part of
          the first meta graph in the SavedModel.
      clear_devices: Set to true if the device info on the default graph should
        be cleared.
      init_op: Op or group of ops to execute when the graph is loaded. Note
          that when the init_op is specified it is run after the restore op at
        load-time.
      train_op: Op or group of opts that trains the model when run. This will
        not be run automatically when the graph is loaded, instead saved in
        a SignatureDef accessible through the exported MetaGraph.
      saver: An instance of tf.compat.v1.train.Saver that will be used to export
        the metagraph. If None, a sharded Saver that restores all variables will
        be used.

    Raises:
      AssertionError: If the variables for the SavedModel have not been saved
          yet, or if the graph already contains one or more legacy init ops.
    """
    ...
  
  def add_meta_graph_and_variables(self, sess, tags, signature_def_map=..., assets_list=..., clear_devices=..., init_op=..., train_op=..., strip_default_attrs=..., saver=...): # -> None:
    """Adds the current meta graph to the SavedModel and saves variables.

    Creates a Saver to save the variables from the provided session. Exports the
    corresponding meta graph def. This function assumes that the variables to be
    saved have been initialized. For a given `SavedModelBuilder`, this API must
    be called exactly once and for the first meta graph to save. For subsequent
    meta graph defs to be added, the `add_meta_graph()` API must be used.

    Args:
      sess: The TensorFlow session from which to save the meta graph and
        variables.
      tags: The set of tags with which to save the meta graph.
      signature_def_map: The map of signature def map to add to the meta graph
        def.
      assets_list: Assets to be saved with SavedModel.
      clear_devices: Set to true if the device info on the default graph should
        be cleared.
      init_op: Op or group of ops to execute when the graph is loaded. Note
          that when the init_op is specified it is run after the restore op at
        load-time.
      train_op: Op or group of ops that trains the model when run. This will
        not be run automatically when the graph is loaded, instead saved in
        a SignatureDef accessible through the exported MetaGraph.
      strip_default_attrs: Boolean. If `True`, default-valued attributes will be
        removed from the NodeDefs. For a detailed guide, see
        [Stripping Default-Valued Attributes](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md#stripping-default-valued-attributes).
      saver: An instance of tf.compat.v1.train.Saver that will be used to export the
        metagraph and save variables. If None, a sharded Saver that restores
        all variables will be used.

    """
    ...
  
  def save(self, as_text=..., experimental_image_format=..., experimental_image_writer_options=...):
    """Writes a `SavedModel` protocol buffer to disk.

    The function writes the SavedModel protocol buffer to the export directory
    in a serialized format.

    Args:
      as_text: Writes the SavedModel protocol buffer in text format to disk.
        Protocol buffers in text format are useful for debugging, but parsing
        fails when it encounters an unknown field and so is not forward
        compatible. This means changes to TensorFlow may prevent deployment of
        new text format SavedModels to existing serving binaries. Do not deploy
        `as_text` SavedModels to production.
      experimental_image_format: Writes the SavedModel protobuf in the
        experimental image format. See
      https://www.tensorflow.org/api_docs/python/tf/saved_model/SaveOptions for
        more details. This allows `SavedModelBuilder` to save models larger than
        2 GiB.
      experimental_image_writer_options: Optional options for the experimental
        image writer. See
        https://github.com/google/riegeli/blob/master/doc/record_writer_options.md
          for available options.

    Raises:
       RuntimeError: When trying to use `proto_splitter` but `proto_splitter` is
         not imported. This check is here because `proto_splitter` is not
         available in OSS at the moment.

    Returns:
      The path to which the SavedModel protocol buffer was written.
    """
    ...
  


@tf_export(v1=["saved_model.Builder", "saved_model.builder.SavedModelBuilder"])
class SavedModelBuilder(_SavedModelBuilder):
  __doc__ = ...
  def __init__(self, export_dir) -> None:
    ...
  
  @deprecated_args(None, "Pass your op to the equivalent parameter main_op instead.", "legacy_init_op")
  def add_meta_graph(self, tags, signature_def_map=..., assets_collection=..., legacy_init_op=..., clear_devices=..., main_op=..., strip_default_attrs=..., saver=...): # -> None:
    ...
  
  @deprecated_args(None, "Pass your op to the equivalent parameter main_op instead.", "legacy_init_op")
  def add_meta_graph_and_variables(self, sess, tags, signature_def_map=..., assets_collection=..., legacy_init_op=..., clear_devices=..., main_op=..., strip_default_attrs=..., saver=...): # -> None:
    ...
  


def get_asset_filename_to_add(asset_filepath, asset_filename_map): # -> bytes:
  """Get a unique basename to add to the SavedModel if this file is unseen.

  Assets come from users as full paths, and we save them out to the
  SavedModel as basenames. In some cases, the basenames collide. Here,
  we dedupe asset basenames by first checking if the file is the same,
  and, if different, generate and return an index-suffixed basename
  that can be used to add the asset to the SavedModel.

  Args:
    asset_filepath: the full path to the asset that is being saved
    asset_filename_map: a dict of filenames used for saving the asset in
      the SavedModel to full paths from which the filenames were derived.

  Returns:
    Uniquified filename string if the file is not a duplicate, or the original
    filename if the file has already been seen and saved.
  """
  ...

def copy_assets_to_destination_dir(asset_filename_map, destination_dir, saved_files=...): # -> None:
  """Copy all assets from source path to destination path.

  Args:
    asset_filename_map: a dict of filenames used for saving the asset in
      the SavedModel to full paths from which the filenames were derived.
    destination_dir: the destination directory that assets are stored in.
    saved_files: a set of destination filepaths that have already been copied
      and will be skipped
  """
  ...

