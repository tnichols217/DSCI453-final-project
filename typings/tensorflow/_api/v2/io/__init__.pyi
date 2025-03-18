"""
This type stub file was generated by pyright.
"""

import sys as _sys
from tensorflow._api.v2.io import gfile
from tensorflow.python.ops.gen_decode_proto_ops import decode_proto_v2 as decode_proto
from tensorflow.python.ops.gen_encode_proto_ops import encode_proto
from tensorflow.python.ops.gen_io_ops import matching_files, write_file
from tensorflow.python.ops.gen_parsing_ops import decode_compressed, parse_tensor
from tensorflow.python.ops.gen_string_ops import decode_base64, encode_base64
from tensorflow.python.framework.graph_io import write_graph
from tensorflow.python.lib.io.tf_record import TFRecordOptions, TFRecordWriter
from tensorflow.python.ops.image_ops_impl import decode_and_crop_jpeg, decode_bmp, decode_gif, decode_image, decode_jpeg, decode_png, encode_jpeg, encode_png, extract_jpeg_shape, is_jpeg
from tensorflow.python.ops.io_ops import read_file, serialize_tensor
from tensorflow.python.ops.parsing_config import FixedLenFeature, FixedLenSequenceFeature, RaggedFeature, SparseFeature, VarLenFeature
from tensorflow.python.ops.parsing_ops import decode_csv_v2 as decode_csv, decode_json_example, decode_raw, parse_example_v2 as parse_example, parse_sequence_example, parse_single_example_v2 as parse_single_example, parse_single_sequence_example
from tensorflow.python.ops.sparse_ops import deserialize_many_sparse, serialize_many_sparse_v2 as serialize_many_sparse, serialize_sparse_v2 as serialize_sparse
from tensorflow.python.training.input import match_filenames_once

"""Public API for tf._api.v2.io namespace
"""
