"""
This type stub file was generated by pyright.
"""

"""Classes and functions that help to inspect Python source w.r.t. TF graphs."""
_TENSORFLOW_BASEDIR = ...
_ABSL_BASEDIR = ...
UNCOMPILED_SOURCE_SUFFIXES = ...
COMPILED_SOURCE_SUFFIXES = ...
def is_extension_uncompiled_python_source(file_path): # -> bool:
  ...

def is_extension_compiled_python_source(file_path): # -> bool:
  ...

def guess_is_tensorflow_py_library(py_file_path): # -> bool:
  """Guess whether a Python source file is a part of the tensorflow library.

  Special cases:
    1) Returns False for unit-test files in the library (*_test.py),
    2) Returns False for files under python/debug/examples.

  Args:
    py_file_path: full path of the Python source file in question.

  Returns:
    (`bool`) Whether the file is inferred to be a part of the tensorflow
      library.
  """
  ...

def load_source(source_file_path): # -> tuple[list[str], int]:
  """Load the content of a Python source code file.

  This function covers the following case:
    1. source_file_path points to an existing Python (.py) file on the
       file system.
    2. source_file_path is a path within a .par file (i.e., a zip-compressed,
       self-contained Python executable).

  Args:
    source_file_path: Path to the Python source file to read.

  Returns:
    A length-2 tuple:
      - Lines of the source file, as a `list` of `str`s.
      - The width of the string needed to show the line number in the file.
        This is calculated based on the number of lines in the source file.

  Raises:
    IOError: if loading is unsuccessful.
  """
  ...

def annotate_source(dump, source_file_path, do_dumped_tensors=..., file_stack_top=..., min_line=..., max_line=...): # -> dict[Any, Any]:
  """Annotate a Python source file with a list of ops created at each line.

  (The annotation doesn't change the source file itself.)

  Args:
    dump: (`DebugDumpDir`) A `DebugDumpDir` object of which the Python graph
      has been loaded.
    source_file_path: (`str`) Path to the source file being annotated.
    do_dumped_tensors: (`str`) Whether dumped Tensors, instead of ops are to be
      used to annotate the source file.
    file_stack_top: (`bool`) Whether only the top stack trace in the
      specified source file is to be annotated.
    min_line: (`None` or `int`) The 1-based line to start annotate the source
      file from (inclusive).
    max_line: (`None` or `int`) The 1-based line number to end the annotation
      at (exclusive).

  Returns:
    A `dict` mapping 1-based line number to a list of op name(s) created at
      that line, or tensor names if `do_dumped_tensors` is True.

  Raises:
    ValueError: If the dump object does not have a Python graph set.
  """
  ...

def list_source_files_against_dump(dump, path_regex_allowlist=..., node_name_regex_allowlist=...): # -> list[Any]:
  """Generate a list of source files with information regarding ops and tensors.

  Args:
    dump: (`DebugDumpDir`) A `DebugDumpDir` object of which the Python graph
      has been loaded.
    path_regex_allowlist: A regular-expression filter for source file path.
    node_name_regex_allowlist: A regular-expression filter for node names.

  Returns:
    A list of tuples regarding the Python source files involved in constructing
    the ops and tensors contained in `dump`. Each tuple is:
      (source_file_path, is_tf_library, num_nodes, num_tensors, num_dumps,
       first_line)

      is_tf_library: (`bool`) A guess of whether the file belongs to the
        TensorFlow Python library.
      num_nodes: How many nodes were created by lines of this source file.
        These include nodes with dumps and those without.
      num_tensors: How many Tensors were created by lines of this source file.
        These include Tensors with dumps and those without.
      num_dumps: How many debug Tensor dumps were from nodes (and Tensors)
        that were created by this source file.
      first_line: The first line number (1-based) that created any nodes or
        Tensors in this source file.

    The list is sorted by ascending order of source_file_path.

  Raises:
    ValueError: If the dump object does not have a Python graph set.
  """
  ...

def annotate_source_against_profile(profile_data, source_file_path, node_name_filter=..., op_type_filter=..., min_line=..., max_line=...): # -> dict[Any, Any]:
  """Annotate a Python source file with profiling information at each line.

  (The annotation doesn't change the source file itself.)

  Args:
    profile_data: (`list` of `ProfileDatum`) A list of `ProfileDatum`.
    source_file_path: (`str`) Path to the source file being annotated.
    node_name_filter: Regular expression to filter by node name.
    op_type_filter: Regular expression to filter by op type.
    min_line: (`None` or `int`) The 1-based line to start annotate the source
      file from (inclusive).
    max_line: (`None` or `int`) The 1-based line number to end the annotation
      at (exclusive).

  Returns:
    A `dict` mapping 1-based line number to a the namedtuple
      `profiling.LineOrFuncProfileSummary`.
  """
  ...

