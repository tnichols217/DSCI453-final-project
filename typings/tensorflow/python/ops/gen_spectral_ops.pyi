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
def batch_fft(input: Annotated[Any, _atypes.Complex64], name=...) -> Annotated[Any, _atypes.Complex64]:
  r"""TODO: add doc.

  Args:
    input: A `Tensor` of type `complex64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `complex64`.
  """
  ...

BatchFFT = ...
def batch_fft_eager_fallback(input: Annotated[Any, _atypes.Complex64], name, ctx) -> Annotated[Any, _atypes.Complex64]:
  ...

def batch_fft2d(input: Annotated[Any, _atypes.Complex64], name=...) -> Annotated[Any, _atypes.Complex64]:
  r"""TODO: add doc.

  Args:
    input: A `Tensor` of type `complex64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `complex64`.
  """
  ...

BatchFFT2D = ...
def batch_fft2d_eager_fallback(input: Annotated[Any, _atypes.Complex64], name, ctx) -> Annotated[Any, _atypes.Complex64]:
  ...

def batch_fft3d(input: Annotated[Any, _atypes.Complex64], name=...) -> Annotated[Any, _atypes.Complex64]:
  r"""TODO: add doc.

  Args:
    input: A `Tensor` of type `complex64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `complex64`.
  """
  ...

BatchFFT3D = ...
def batch_fft3d_eager_fallback(input: Annotated[Any, _atypes.Complex64], name, ctx) -> Annotated[Any, _atypes.Complex64]:
  ...

def batch_ifft(input: Annotated[Any, _atypes.Complex64], name=...) -> Annotated[Any, _atypes.Complex64]:
  r"""TODO: add doc.

  Args:
    input: A `Tensor` of type `complex64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `complex64`.
  """
  ...

BatchIFFT = ...
def batch_ifft_eager_fallback(input: Annotated[Any, _atypes.Complex64], name, ctx) -> Annotated[Any, _atypes.Complex64]:
  ...

def batch_ifft2d(input: Annotated[Any, _atypes.Complex64], name=...) -> Annotated[Any, _atypes.Complex64]:
  r"""TODO: add doc.

  Args:
    input: A `Tensor` of type `complex64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `complex64`.
  """
  ...

BatchIFFT2D = ...
def batch_ifft2d_eager_fallback(input: Annotated[Any, _atypes.Complex64], name, ctx) -> Annotated[Any, _atypes.Complex64]:
  ...

def batch_ifft3d(input: Annotated[Any, _atypes.Complex64], name=...) -> Annotated[Any, _atypes.Complex64]:
  r"""TODO: add doc.

  Args:
    input: A `Tensor` of type `complex64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `complex64`.
  """
  ...

BatchIFFT3D = ...
def batch_ifft3d_eager_fallback(input: Annotated[Any, _atypes.Complex64], name, ctx) -> Annotated[Any, _atypes.Complex64]:
  ...

TV_FFT_Tcomplex = TypeVar("TV_FFT_Tcomplex", _atypes.Complex128, _atypes.Complex64)
@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('signal.fft', v1=['signal.fft', 'spectral.fft', 'fft'])
@deprecated_endpoints('spectral.fft', 'fft')
def fft(input: Annotated[Any, TV_FFT_Tcomplex], name=...) -> Annotated[Any, TV_FFT_Tcomplex]:
  r"""Fast Fourier transform.

  Computes the 1-dimensional discrete Fourier transform over the inner-most
  dimension of `input`.

  Args:
    input: A `Tensor`. Must be one of the following types: `complex64`, `complex128`.
      A complex tensor.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  ...

FFT = ...
_dispatcher_for_fft = fft._tf_type_based_dispatcher.Dispatch
def fft_eager_fallback(input: Annotated[Any, TV_FFT_Tcomplex], name, ctx) -> Annotated[Any, TV_FFT_Tcomplex]:
  ...

TV_FFT2D_Tcomplex = TypeVar("TV_FFT2D_Tcomplex", _atypes.Complex128, _atypes.Complex64)
@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('signal.fft2d', v1=['signal.fft2d', 'spectral.fft2d', 'fft2d'])
@deprecated_endpoints('spectral.fft2d', 'fft2d')
def fft2d(input: Annotated[Any, TV_FFT2D_Tcomplex], name=...) -> Annotated[Any, TV_FFT2D_Tcomplex]:
  r"""2D fast Fourier transform.

  Computes the 2-dimensional discrete Fourier transform over the inner-most
  2 dimensions of `input`.

  Args:
    input: A `Tensor`. Must be one of the following types: `complex64`, `complex128`.
      A complex tensor.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  ...

FFT2D = ...
_dispatcher_for_fft2d = fft2d._tf_type_based_dispatcher.Dispatch
def fft2d_eager_fallback(input: Annotated[Any, TV_FFT2D_Tcomplex], name, ctx) -> Annotated[Any, TV_FFT2D_Tcomplex]:
  ...

TV_FFT3D_Tcomplex = TypeVar("TV_FFT3D_Tcomplex", _atypes.Complex128, _atypes.Complex64)
@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('signal.fft3d', v1=['signal.fft3d', 'spectral.fft3d', 'fft3d'])
@deprecated_endpoints('spectral.fft3d', 'fft3d')
def fft3d(input: Annotated[Any, TV_FFT3D_Tcomplex], name=...) -> Annotated[Any, TV_FFT3D_Tcomplex]:
  r"""3D fast Fourier transform.

  Computes the 3-dimensional discrete Fourier transform over the inner-most 3
  dimensions of `input`.

  Args:
    input: A `Tensor`. Must be one of the following types: `complex64`, `complex128`.
      A complex tensor.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  ...

FFT3D = ...
_dispatcher_for_fft3d = fft3d._tf_type_based_dispatcher.Dispatch
def fft3d_eager_fallback(input: Annotated[Any, TV_FFT3D_Tcomplex], name, ctx) -> Annotated[Any, TV_FFT3D_Tcomplex]:
  ...

TV_FFTND_Tcomplex = TypeVar("TV_FFTND_Tcomplex", _atypes.Complex128, _atypes.Complex64)
@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('fftnd')
def fftnd(input: Annotated[Any, TV_FFTND_Tcomplex], fft_length: Annotated[Any, _atypes.Int32], axes: Annotated[Any, _atypes.Int32], name=...) -> Annotated[Any, TV_FFTND_Tcomplex]:
  r"""ND fast Fourier transform.

  Computes the n-dimensional discrete Fourier transform over
  designated dimensions of `input`. The designated dimensions of
  `input` are assumed to be the result of `FFTND`.

  If fft_length[i]<shape(input)[i], the input is cropped. If
  fft_length[i]>shape(input)[i], the input is padded with zeros. If fft_length
  is not given, the default shape(input) is used.

  Axes mean the dimensions to perform the transform on. Default is to perform on
  all axes.

  Args:
    input: A `Tensor`. Must be one of the following types: `complex64`, `complex128`.
      A complex tensor.
    fft_length: A `Tensor` of type `int32`.
      An int32 tensor. The FFT length for each dimension.
    axes: A `Tensor` of type `int32`.
      An int32 tensor with a same shape as fft_length. Axes to perform the transform.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  ...

FFTND = ...
_dispatcher_for_fftnd = fftnd._tf_type_based_dispatcher.Dispatch
def fftnd_eager_fallback(input: Annotated[Any, TV_FFTND_Tcomplex], fft_length: Annotated[Any, _atypes.Int32], axes: Annotated[Any, _atypes.Int32], name, ctx) -> Annotated[Any, TV_FFTND_Tcomplex]:
  ...

TV_IFFT_Tcomplex = TypeVar("TV_IFFT_Tcomplex", _atypes.Complex128, _atypes.Complex64)
@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('signal.ifft', v1=['signal.ifft', 'spectral.ifft', 'ifft'])
@deprecated_endpoints('spectral.ifft', 'ifft')
def ifft(input: Annotated[Any, TV_IFFT_Tcomplex], name=...) -> Annotated[Any, TV_IFFT_Tcomplex]:
  r"""Inverse fast Fourier transform.

  Computes the inverse 1-dimensional discrete Fourier transform over the
  inner-most dimension of `input`.

  Args:
    input: A `Tensor`. Must be one of the following types: `complex64`, `complex128`.
      A complex tensor.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  ...

IFFT = ...
_dispatcher_for_ifft = ifft._tf_type_based_dispatcher.Dispatch
def ifft_eager_fallback(input: Annotated[Any, TV_IFFT_Tcomplex], name, ctx) -> Annotated[Any, TV_IFFT_Tcomplex]:
  ...

TV_IFFT2D_Tcomplex = TypeVar("TV_IFFT2D_Tcomplex", _atypes.Complex128, _atypes.Complex64)
@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('signal.ifft2d', v1=['signal.ifft2d', 'spectral.ifft2d', 'ifft2d'])
@deprecated_endpoints('spectral.ifft2d', 'ifft2d')
def ifft2d(input: Annotated[Any, TV_IFFT2D_Tcomplex], name=...) -> Annotated[Any, TV_IFFT2D_Tcomplex]:
  r"""Inverse 2D fast Fourier transform.

  Computes the inverse 2-dimensional discrete Fourier transform over the
  inner-most 2 dimensions of `input`.

  Args:
    input: A `Tensor`. Must be one of the following types: `complex64`, `complex128`.
      A complex tensor.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  ...

IFFT2D = ...
_dispatcher_for_ifft2d = ifft2d._tf_type_based_dispatcher.Dispatch
def ifft2d_eager_fallback(input: Annotated[Any, TV_IFFT2D_Tcomplex], name, ctx) -> Annotated[Any, TV_IFFT2D_Tcomplex]:
  ...

TV_IFFT3D_Tcomplex = TypeVar("TV_IFFT3D_Tcomplex", _atypes.Complex128, _atypes.Complex64)
@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('signal.ifft3d', v1=['signal.ifft3d', 'spectral.ifft3d', 'ifft3d'])
@deprecated_endpoints('spectral.ifft3d', 'ifft3d')
def ifft3d(input: Annotated[Any, TV_IFFT3D_Tcomplex], name=...) -> Annotated[Any, TV_IFFT3D_Tcomplex]:
  r"""Inverse 3D fast Fourier transform.

  Computes the inverse 3-dimensional discrete Fourier transform over the
  inner-most 3 dimensions of `input`.

  Args:
    input: A `Tensor`. Must be one of the following types: `complex64`, `complex128`.
      A complex tensor.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  ...

IFFT3D = ...
_dispatcher_for_ifft3d = ifft3d._tf_type_based_dispatcher.Dispatch
def ifft3d_eager_fallback(input: Annotated[Any, TV_IFFT3D_Tcomplex], name, ctx) -> Annotated[Any, TV_IFFT3D_Tcomplex]:
  ...

TV_IFFTND_Tcomplex = TypeVar("TV_IFFTND_Tcomplex", _atypes.Complex128, _atypes.Complex64)
@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('ifftnd')
def ifftnd(input: Annotated[Any, TV_IFFTND_Tcomplex], fft_length: Annotated[Any, _atypes.Int32], axes: Annotated[Any, _atypes.Int32], name=...) -> Annotated[Any, TV_IFFTND_Tcomplex]:
  r"""ND inverse fast Fourier transform.

  Computes the n-dimensional inverse discrete Fourier transform over designated
  dimensions of `input`. The designated dimensions of `input` are assumed to be
  the result of `IFFTND`.

  If fft_length[i]<shape(input)[i], the input is cropped. If
  fft_length[i]>shape(input)[i], the input is padded with zeros. If fft_length
  is not given, the default shape(input) is used.

  Axes mean the dimensions to perform the transform on. Default is to perform on
  all axes.

  Args:
    input: A `Tensor`. Must be one of the following types: `complex64`, `complex128`.
      A complex tensor.
    fft_length: A `Tensor` of type `int32`.
      An int32 tensor. The FFT length for each dimension.
    axes: A `Tensor` of type `int32`.
      An int32 tensor with a same shape as fft_length. Axes to perform the transform.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  ...

IFFTND = ...
_dispatcher_for_ifftnd = ifftnd._tf_type_based_dispatcher.Dispatch
def ifftnd_eager_fallback(input: Annotated[Any, TV_IFFTND_Tcomplex], fft_length: Annotated[Any, _atypes.Int32], axes: Annotated[Any, _atypes.Int32], name, ctx) -> Annotated[Any, TV_IFFTND_Tcomplex]:
  ...

TV_IRFFT_Treal = TypeVar("TV_IRFFT_Treal", _atypes.Float32, _atypes.Float64)
TV_IRFFT_Tcomplex = TypeVar("TV_IRFFT_Tcomplex", _atypes.Complex128, _atypes.Complex64)
def irfft(input: Annotated[Any, TV_IRFFT_Tcomplex], fft_length: Annotated[Any, _atypes.Int32], Treal: TV_IRFFT_Treal = ..., name=...) -> Annotated[Any, TV_IRFFT_Treal]:
  r"""Inverse real-valued fast Fourier transform.

  Computes the inverse 1-dimensional discrete Fourier transform of a real-valued
  signal over the inner-most dimension of `input`.

  The inner-most dimension of `input` is assumed to be the result of `RFFT`: the
  `fft_length / 2 + 1` unique components of the DFT of a real-valued signal. If
  `fft_length` is not provided, it is computed from the size of the inner-most
  dimension of `input` (`fft_length = 2 * (inner - 1)`). If the FFT length used to
  compute `input` is odd, it should be provided since it cannot be inferred
  properly.

  Along the axis `IRFFT` is computed on, if `fft_length / 2 + 1` is smaller
  than the corresponding dimension of `input`, the dimension is cropped. If it is
  larger, the dimension is padded with zeros.

  Args:
    input: A `Tensor`. Must be one of the following types: `complex64`, `complex128`.
      A complex tensor.
    fft_length: A `Tensor` of type `int32`.
      An int32 tensor of shape [1]. The FFT length.
    Treal: An optional `tf.DType` from: `tf.float32, tf.float64`. Defaults to `tf.float32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `Treal`.
  """
  ...

IRFFT = ...
def irfft_eager_fallback(input: Annotated[Any, TV_IRFFT_Tcomplex], fft_length: Annotated[Any, _atypes.Int32], Treal: TV_IRFFT_Treal, name, ctx) -> Annotated[Any, TV_IRFFT_Treal]:
  ...

TV_IRFFT2D_Treal = TypeVar("TV_IRFFT2D_Treal", _atypes.Float32, _atypes.Float64)
TV_IRFFT2D_Tcomplex = TypeVar("TV_IRFFT2D_Tcomplex", _atypes.Complex128, _atypes.Complex64)
def irfft2d(input: Annotated[Any, TV_IRFFT2D_Tcomplex], fft_length: Annotated[Any, _atypes.Int32], Treal: TV_IRFFT2D_Treal = ..., name=...) -> Annotated[Any, TV_IRFFT2D_Treal]:
  r"""Inverse 2D real-valued fast Fourier transform.

  Computes the inverse 2-dimensional discrete Fourier transform of a real-valued
  signal over the inner-most 2 dimensions of `input`.

  The inner-most 2 dimensions of `input` are assumed to be the result of `RFFT2D`:
  The inner-most dimension contains the `fft_length / 2 + 1` unique components of
  the DFT of a real-valued signal. If `fft_length` is not provided, it is computed
  from the size of the inner-most 2 dimensions of `input`. If the FFT length used
  to compute `input` is odd, it should be provided since it cannot be inferred
  properly.

  Along each axis `IRFFT2D` is computed on, if `fft_length` (or
  `fft_length / 2 + 1` for the inner-most dimension) is smaller than the
  corresponding dimension of `input`, the dimension is cropped. If it is larger,
  the dimension is padded with zeros.

  Args:
    input: A `Tensor`. Must be one of the following types: `complex64`, `complex128`.
      A complex tensor.
    fft_length: A `Tensor` of type `int32`.
      An int32 tensor of shape [2]. The FFT length for each dimension.
    Treal: An optional `tf.DType` from: `tf.float32, tf.float64`. Defaults to `tf.float32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `Treal`.
  """
  ...

IRFFT2D = ...
def irfft2d_eager_fallback(input: Annotated[Any, TV_IRFFT2D_Tcomplex], fft_length: Annotated[Any, _atypes.Int32], Treal: TV_IRFFT2D_Treal, name, ctx) -> Annotated[Any, TV_IRFFT2D_Treal]:
  ...

TV_IRFFT3D_Treal = TypeVar("TV_IRFFT3D_Treal", _atypes.Float32, _atypes.Float64)
TV_IRFFT3D_Tcomplex = TypeVar("TV_IRFFT3D_Tcomplex", _atypes.Complex128, _atypes.Complex64)
def irfft3d(input: Annotated[Any, TV_IRFFT3D_Tcomplex], fft_length: Annotated[Any, _atypes.Int32], Treal: TV_IRFFT3D_Treal = ..., name=...) -> Annotated[Any, TV_IRFFT3D_Treal]:
  r"""Inverse 3D real-valued fast Fourier transform.

  Computes the inverse 3-dimensional discrete Fourier transform of a real-valued
  signal over the inner-most 3 dimensions of `input`.

  The inner-most 3 dimensions of `input` are assumed to be the result of `RFFT3D`:
  The inner-most dimension contains the `fft_length / 2 + 1` unique components of
  the DFT of a real-valued signal. If `fft_length` is not provided, it is computed
  from the size of the inner-most 3 dimensions of `input`. If the FFT length used
  to compute `input` is odd, it should be provided since it cannot be inferred
  properly.

  Along each axis `IRFFT3D` is computed on, if `fft_length` (or
  `fft_length / 2 + 1` for the inner-most dimension) is smaller than the
  corresponding dimension of `input`, the dimension is cropped. If it is larger,
  the dimension is padded with zeros.

  Args:
    input: A `Tensor`. Must be one of the following types: `complex64`, `complex128`.
      A complex tensor.
    fft_length: A `Tensor` of type `int32`.
      An int32 tensor of shape [3]. The FFT length for each dimension.
    Treal: An optional `tf.DType` from: `tf.float32, tf.float64`. Defaults to `tf.float32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `Treal`.
  """
  ...

IRFFT3D = ...
def irfft3d_eager_fallback(input: Annotated[Any, TV_IRFFT3D_Tcomplex], fft_length: Annotated[Any, _atypes.Int32], Treal: TV_IRFFT3D_Treal, name, ctx) -> Annotated[Any, TV_IRFFT3D_Treal]:
  ...

TV_IRFFTND_Treal = TypeVar("TV_IRFFTND_Treal", _atypes.Float32, _atypes.Float64)
TV_IRFFTND_Tcomplex = TypeVar("TV_IRFFTND_Tcomplex", _atypes.Complex128, _atypes.Complex64)
@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('irfftnd')
def irfftnd(input: Annotated[Any, TV_IRFFTND_Tcomplex], fft_length: Annotated[Any, _atypes.Int32], axes: Annotated[Any, _atypes.Int32], Treal: TV_IRFFTND_Treal = ..., name=...) -> Annotated[Any, TV_IRFFTND_Treal]:
  r"""ND inverse real fast Fourier transform.

  Computes the n-dimensional inverse real discrete Fourier transform over
  designated dimensions of `input`. The designated dimensions of `input` are
  assumed to be the result of `IRFFTND`. The inner-most dimension contains the
  `fft_length / 2 + 1` unique components of the DFT of a real-valued signal. 

  If fft_length[i]<shape(input)[i], the input is cropped. If
  fft_length[i]>shape(input)[i], the input is padded with zeros. If fft_length
  is not given, the default shape(input) is used.

  Axes mean the dimensions to perform the transform on. Default is to perform on
  all axes.

  Args:
    input: A `Tensor`. Must be one of the following types: `complex64`, `complex128`.
      A complex tensor.
    fft_length: A `Tensor` of type `int32`.
      An int32 tensor. The FFT length for each dimension.
    axes: A `Tensor` of type `int32`.
      An int32 tensor with a same shape as fft_length. Axes to perform the transform.
    Treal: An optional `tf.DType` from: `tf.float32, tf.float64`. Defaults to `tf.float32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `Treal`.
  """
  ...

IRFFTND = ...
_dispatcher_for_irfftnd = irfftnd._tf_type_based_dispatcher.Dispatch
def irfftnd_eager_fallback(input: Annotated[Any, TV_IRFFTND_Tcomplex], fft_length: Annotated[Any, _atypes.Int32], axes: Annotated[Any, _atypes.Int32], Treal: TV_IRFFTND_Treal, name, ctx) -> Annotated[Any, TV_IRFFTND_Treal]:
  ...

TV_RFFT_Treal = TypeVar("TV_RFFT_Treal", _atypes.Float32, _atypes.Float64)
TV_RFFT_Tcomplex = TypeVar("TV_RFFT_Tcomplex", _atypes.Complex128, _atypes.Complex64)
def rfft(input: Annotated[Any, TV_RFFT_Treal], fft_length: Annotated[Any, _atypes.Int32], Tcomplex: TV_RFFT_Tcomplex = ..., name=...) -> Annotated[Any, TV_RFFT_Tcomplex]:
  r"""Real-valued fast Fourier transform.

  Computes the 1-dimensional discrete Fourier transform of a real-valued signal
  over the inner-most dimension of `input`.

  Since the DFT of a real signal is Hermitian-symmetric, `RFFT` only returns the
  `fft_length / 2 + 1` unique components of the FFT: the zero-frequency term,
  followed by the `fft_length / 2` positive-frequency terms.

  Along the axis `RFFT` is computed on, if `fft_length` is smaller than the
  corresponding dimension of `input`, the dimension is cropped. If it is larger,
  the dimension is padded with zeros.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`.
      A float32 tensor.
    fft_length: A `Tensor` of type `int32`.
      An int32 tensor of shape [1]. The FFT length.
    Tcomplex: An optional `tf.DType` from: `tf.complex64, tf.complex128`. Defaults to `tf.complex64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `Tcomplex`.
  """
  ...

RFFT = ...
def rfft_eager_fallback(input: Annotated[Any, TV_RFFT_Treal], fft_length: Annotated[Any, _atypes.Int32], Tcomplex: TV_RFFT_Tcomplex, name, ctx) -> Annotated[Any, TV_RFFT_Tcomplex]:
  ...

TV_RFFT2D_Treal = TypeVar("TV_RFFT2D_Treal", _atypes.Float32, _atypes.Float64)
TV_RFFT2D_Tcomplex = TypeVar("TV_RFFT2D_Tcomplex", _atypes.Complex128, _atypes.Complex64)
def rfft2d(input: Annotated[Any, TV_RFFT2D_Treal], fft_length: Annotated[Any, _atypes.Int32], Tcomplex: TV_RFFT2D_Tcomplex = ..., name=...) -> Annotated[Any, TV_RFFT2D_Tcomplex]:
  r"""2D real-valued fast Fourier transform.

  Computes the 2-dimensional discrete Fourier transform of a real-valued signal
  over the inner-most 2 dimensions of `input`.

  Since the DFT of a real signal is Hermitian-symmetric, `RFFT2D` only returns the
  `fft_length / 2 + 1` unique components of the FFT for the inner-most dimension
  of `output`: the zero-frequency term, followed by the `fft_length / 2`
  positive-frequency terms.

  Along each axis `RFFT2D` is computed on, if `fft_length` is smaller than the
  corresponding dimension of `input`, the dimension is cropped. If it is larger,
  the dimension is padded with zeros.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`.
      A float32 tensor.
    fft_length: A `Tensor` of type `int32`.
      An int32 tensor of shape [2]. The FFT length for each dimension.
    Tcomplex: An optional `tf.DType` from: `tf.complex64, tf.complex128`. Defaults to `tf.complex64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `Tcomplex`.
  """
  ...

RFFT2D = ...
def rfft2d_eager_fallback(input: Annotated[Any, TV_RFFT2D_Treal], fft_length: Annotated[Any, _atypes.Int32], Tcomplex: TV_RFFT2D_Tcomplex, name, ctx) -> Annotated[Any, TV_RFFT2D_Tcomplex]:
  ...

TV_RFFT3D_Treal = TypeVar("TV_RFFT3D_Treal", _atypes.Float32, _atypes.Float64)
TV_RFFT3D_Tcomplex = TypeVar("TV_RFFT3D_Tcomplex", _atypes.Complex128, _atypes.Complex64)
def rfft3d(input: Annotated[Any, TV_RFFT3D_Treal], fft_length: Annotated[Any, _atypes.Int32], Tcomplex: TV_RFFT3D_Tcomplex = ..., name=...) -> Annotated[Any, TV_RFFT3D_Tcomplex]:
  r"""3D real-valued fast Fourier transform.

  Computes the 3-dimensional discrete Fourier transform of a real-valued signal
  over the inner-most 3 dimensions of `input`.

  Since the DFT of a real signal is Hermitian-symmetric, `RFFT3D` only returns the
  `fft_length / 2 + 1` unique components of the FFT for the inner-most dimension
  of `output`: the zero-frequency term, followed by the `fft_length / 2`
  positive-frequency terms.

  Along each axis `RFFT3D` is computed on, if `fft_length` is smaller than the
  corresponding dimension of `input`, the dimension is cropped. If it is larger,
  the dimension is padded with zeros.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`.
      A float32 tensor.
    fft_length: A `Tensor` of type `int32`.
      An int32 tensor of shape [3]. The FFT length for each dimension.
    Tcomplex: An optional `tf.DType` from: `tf.complex64, tf.complex128`. Defaults to `tf.complex64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `Tcomplex`.
  """
  ...

RFFT3D = ...
def rfft3d_eager_fallback(input: Annotated[Any, TV_RFFT3D_Treal], fft_length: Annotated[Any, _atypes.Int32], Tcomplex: TV_RFFT3D_Tcomplex, name, ctx) -> Annotated[Any, TV_RFFT3D_Tcomplex]:
  ...

TV_RFFTND_Treal = TypeVar("TV_RFFTND_Treal", _atypes.Float32, _atypes.Float64)
TV_RFFTND_Tcomplex = TypeVar("TV_RFFTND_Tcomplex", _atypes.Complex128, _atypes.Complex64)
@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('rfftnd')
def rfftnd(input: Annotated[Any, TV_RFFTND_Treal], fft_length: Annotated[Any, _atypes.Int32], axes: Annotated[Any, _atypes.Int32], Tcomplex: TV_RFFTND_Tcomplex = ..., name=...) -> Annotated[Any, TV_RFFTND_Tcomplex]:
  r"""ND fast real Fourier transform.

  Computes the n-dimensional real discrete Fourier transform over designated
  dimensions of `input`. The designated dimensions of `input` are assumed to be
  the result of `RFFTND`. The length of the last axis transformed will be
  fft_length[-1]//2+1.

  If fft_length[i]<shape(input)[i], the input is cropped. If
  fft_length[i]>shape(input)[i], the input is padded with zeros. If fft_length
  is not given, the default shape(input) is used.

  Axes mean the dimensions to perform the transform on. Default is to perform on
  all axes.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`.
      A complex tensor.
    fft_length: A `Tensor` of type `int32`.
      An int32 tensor. The FFT length for each dimension.
    axes: A `Tensor` of type `int32`.
      An int32 tensor with a same shape as fft_length. Axes to perform the transform.
    Tcomplex: An optional `tf.DType` from: `tf.complex64, tf.complex128`. Defaults to `tf.complex64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `Tcomplex`.
  """
  ...

RFFTND = ...
_dispatcher_for_rfftnd = rfftnd._tf_type_based_dispatcher.Dispatch
def rfftnd_eager_fallback(input: Annotated[Any, TV_RFFTND_Treal], fft_length: Annotated[Any, _atypes.Int32], axes: Annotated[Any, _atypes.Int32], Tcomplex: TV_RFFTND_Tcomplex, name, ctx) -> Annotated[Any, TV_RFFTND_Tcomplex]:
  ...

