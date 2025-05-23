"""
This type stub file was generated by pyright.
"""

import abc
from tensorflow.python.framework import composite_tensor, composite_tensor_gradient, type_spec
from tensorflow.python.module import module
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export

"""Base class for linear operators."""
__all__ = ["LinearOperator"]
class _LinearOperatorGradient(composite_tensor_gradient.CompositeTensorGradient):
  """Composite tensor gradient for `LinearOperator`."""
  def get_gradient_components(self, value):
    ...
  
  def replace_gradient_components(self, value, components): # -> None:
    ...
  


@tf_export("linalg.LinearOperator")
class LinearOperator(module.Module, composite_tensor.CompositeTensor, metaclass=abc.ABCMeta):
  """Base class defining a [batch of] linear operator[s].

  Subclasses of `LinearOperator` provide access to common methods on a
  (batch) matrix, without the need to materialize the matrix.  This allows:

  * Matrix free computations
  * Operators that take advantage of special structure, while providing a
    consistent API to users.

  #### Subclassing

  To enable a public method, subclasses should implement the leading-underscore
  version of the method.  The argument signature should be identical except for
  the omission of `name="..."`.  For example, to enable
  `matmul(x, adjoint=False, name="matmul")` a subclass should implement
  `_matmul(x, adjoint=False)`.

  #### Performance contract

  Subclasses should only implement the assert methods
  (e.g. `assert_non_singular`) if they can be done in less than `O(N^3)`
  time.

  Class docstrings should contain an explanation of computational complexity.
  Since this is a high-performance library, attention should be paid to detail,
  and explanations can include constants as well as Big-O notation.

  #### Shape compatibility

  `LinearOperator` subclasses should operate on a [batch] matrix with
  compatible shape.  Class docstrings should define what is meant by compatible
  shape.  Some subclasses may not support batching.

  Examples:

  `x` is a batch matrix with compatible shape for `matmul` if

  ```
  operator.shape = [B1,...,Bb] + [M, N],  b >= 0,
  x.shape =   [B1,...,Bb] + [N, R]
  ```

  `rhs` is a batch matrix with compatible shape for `solve` if

  ```
  operator.shape = [B1,...,Bb] + [M, N],  b >= 0,
  rhs.shape =   [B1,...,Bb] + [M, R]
  ```

  #### Example docstring for subclasses.

  This operator acts like a (batch) matrix `A` with shape
  `[B1,...,Bb, M, N]` for some `b >= 0`.  The first `b` indices index a
  batch member.  For every batch index `(i1,...,ib)`, `A[i1,...,ib, : :]` is
  an `m x n` matrix.  Again, this matrix `A` may not be materialized, but for
  purposes of identifying and working with compatible arguments the shape is
  relevant.

  Examples:

  ```python
  some_tensor = ... shape = ????
  operator = MyLinOp(some_tensor)

  operator.shape()
  ==> [2, 4, 4]

  operator.log_abs_determinant()
  ==> Shape [2] Tensor

  x = ... Shape [2, 4, 5] Tensor

  operator.matmul(x)
  ==> Shape [2, 4, 5] Tensor
  ```

  #### Shape compatibility

  This operator acts on batch matrices with compatible shape.
  FILL IN WHAT IS MEANT BY COMPATIBLE SHAPE

  #### Performance

  FILL THIS IN

  #### Matrix property hints

  This `LinearOperator` is initialized with boolean flags of the form `is_X`,
  for `X = non_singular, self_adjoint, positive_definite, square`.
  These have the following meaning:

  * If `is_X == True`, callers should expect the operator to have the
    property `X`.  This is a promise that should be fulfilled, but is *not* a
    runtime assert.  For example, finite floating point precision may result
    in these promises being violated.
  * If `is_X == False`, callers should expect the operator to not have `X`.
  * If `is_X == None` (the default), callers should have no expectation either
    way.

  #### Initialization parameters

  All subclasses of `LinearOperator` are expected to pass a `parameters`
  argument to `super().__init__()`.  This should be a `dict` containing
  the unadulterated arguments passed to the subclass `__init__`.  For example,
  `MyLinearOperator` with an initializer should look like:

  ```python
  def __init__(self, operator, is_square=False, name=None):
     parameters = dict(
         operator=operator,
         is_square=is_square,
         name=name
     )
     ...
     super().__init__(..., parameters=parameters)
  ```

   Users can then access `my_linear_operator.parameters` to see all arguments
   passed to its initializer.
  """
  @deprecation.deprecated_args(None, "Do not pass `graph_parents`.  They will " " no longer be used.", "graph_parents")
  def __init__(self, dtype, graph_parents=..., is_non_singular=..., is_self_adjoint=..., is_positive_definite=..., is_square=..., name=..., parameters=...) -> None:
    """Initialize the `LinearOperator`.

    **This is a private method for subclass use.**
    **Subclasses should copy-paste this `__init__` documentation.**

    Args:
      dtype: The type of the this `LinearOperator`.  Arguments to `matmul` and
        `solve` will have to be this type.
      graph_parents: (Deprecated) Python list of graph prerequisites of this
        `LinearOperator` Typically tensors that are passed during initialization
      is_non_singular:  Expect that this operator is non-singular.
      is_self_adjoint:  Expect that this operator is equal to its hermitian
        transpose.  If `dtype` is real, this is equivalent to being symmetric.
      is_positive_definite:  Expect that this operator is positive definite,
        meaning the quadratic form `x^H A x` has positive real part for all
        nonzero `x`.  Note that we do not require the operator to be
        self-adjoint to be positive-definite.  See:
        https://en.wikipedia.org/wiki/Positive-definite_matrix#Extension_for_non-symmetric_matrices
      is_square:  Expect that this operator acts like square [batch] matrices.
      name: A name for this `LinearOperator`.
      parameters: Python `dict` of parameters used to instantiate this
        `LinearOperator`.

    Raises:
      ValueError:  If any member of graph_parents is `None` or not a `Tensor`.
      ValueError:  If hints are set incorrectly.
    """
    ...
  
  @property
  def parameters(self): # -> dict[bytes, bytes]:
    """Dictionary of parameters used to instantiate this `LinearOperator`."""
    ...
  
  @property
  def dtype(self): # -> DType | Any:
    """The `DType` of `Tensor`s handled by this `LinearOperator`."""
    ...
  
  @property
  def name(self): # -> str:
    """Name prepended to all ops created by this `LinearOperator`."""
    ...
  
  @property
  @deprecation.deprecated(None, "Do not call `graph_parents`.")
  def graph_parents(self): # -> list[Any]:
    """List of graph dependencies of this `LinearOperator`."""
    ...
  
  @property
  def is_non_singular(self): # -> bool | None:
    ...
  
  @property
  def is_self_adjoint(self): # -> None:
    ...
  
  @property
  def is_positive_definite(self): # -> None:
    ...
  
  @property
  def is_square(self): # -> _NotImplementedType | bool | None:
    """Return `True/False` depending on if this operator is square."""
    ...
  
  @property
  def shape(self):
    """`TensorShape` of this `LinearOperator`.

    If this operator acts like the batch matrix `A` with
    `A.shape = [B1,...,Bb, M, N]`, then this returns
    `TensorShape([B1,...,Bb, M, N])`, equivalent to `A.shape`.

    Returns:
      `TensorShape`, statically determined, may be undefined.
    """
    ...
  
  def shape_tensor(self, name=...): # -> Tensor:
    """Shape of this `LinearOperator`, determined at runtime.

    If this operator acts like the batch matrix `A` with
    `A.shape = [B1,...,Bb, M, N]`, then this returns a `Tensor` holding
    `[B1,...,Bb, M, N]`, equivalent to `tf.shape(A)`.

    Args:
      name:  A name for this `Op`.

    Returns:
      `int32` `Tensor`
    """
    ...
  
  @property
  def batch_shape(self):
    """`TensorShape` of batch dimensions of this `LinearOperator`.

    If this operator acts like the batch matrix `A` with
    `A.shape = [B1,...,Bb, M, N]`, then this returns
    `TensorShape([B1,...,Bb])`, equivalent to `A.shape[:-2]`

    Returns:
      `TensorShape`, statically determined, may be undefined.
    """
    ...
  
  def batch_shape_tensor(self, name=...): # -> Tensor:
    """Shape of batch dimensions of this operator, determined at runtime.

    If this operator acts like the batch matrix `A` with
    `A.shape = [B1,...,Bb, M, N]`, then this returns a `Tensor` holding
    `[B1,...,Bb]`.

    Args:
      name:  A name for this `Op`.

    Returns:
      `int32` `Tensor`
    """
    ...
  
  @property
  def tensor_rank(self, name=...):
    """Rank (in the sense of tensors) of matrix corresponding to this operator.

    If this operator acts like the batch matrix `A` with
    `A.shape = [B1,...,Bb, M, N]`, then this returns `b + 2`.

    Args:
      name:  A name for this `Op`.

    Returns:
      Python integer, or None if the tensor rank is undefined.
    """
    ...
  
  def tensor_rank_tensor(self, name=...): # -> Tensor | SymbolicTensor | Operation | _EagerTensorBase | Any:
    """Rank (in the sense of tensors) of matrix corresponding to this operator.

    If this operator acts like the batch matrix `A` with
    `A.shape = [B1,...,Bb, M, N]`, then this returns `b + 2`.

    Args:
      name:  A name for this `Op`.

    Returns:
      `int32` `Tensor`, determined at runtime.
    """
    ...
  
  @property
  def domain_dimension(self): # -> Dimension:
    """Dimension (in the sense of vector spaces) of the domain of this operator.

    If this operator acts like the batch matrix `A` with
    `A.shape = [B1,...,Bb, M, N]`, then this returns `N`.

    Returns:
      `Dimension` object.
    """
    ...
  
  def domain_dimension_tensor(self, name=...): # -> Tensor:
    """Dimension (in the sense of vector spaces) of the domain of this operator.

    Determined at runtime.

    If this operator acts like the batch matrix `A` with
    `A.shape = [B1,...,Bb, M, N]`, then this returns `N`.

    Args:
      name:  A name for this `Op`.

    Returns:
      `int32` `Tensor`
    """
    ...
  
  @property
  def range_dimension(self): # -> Dimension:
    """Dimension (in the sense of vector spaces) of the range of this operator.

    If this operator acts like the batch matrix `A` with
    `A.shape = [B1,...,Bb, M, N]`, then this returns `M`.

    Returns:
      `Dimension` object.
    """
    ...
  
  def range_dimension_tensor(self, name=...): # -> Tensor:
    """Dimension (in the sense of vector spaces) of the range of this operator.

    Determined at runtime.

    If this operator acts like the batch matrix `A` with
    `A.shape = [B1,...,Bb, M, N]`, then this returns `M`.

    Args:
      name:  A name for this `Op`.

    Returns:
      `int32` `Tensor`
    """
    ...
  
  def assert_non_singular(self, name=...): # -> object | Operation | Any | None:
    """Returns an `Op` that asserts this operator is non singular.

    This operator is considered non-singular if

    ```
    ConditionNumber < max{100, range_dimension, domain_dimension} * eps,
    eps := np.finfo(self.dtype.as_numpy_dtype).eps
    ```

    Args:
      name:  A string name to prepend to created ops.

    Returns:
      An `Assert` `Op`, that, when run, will raise an `InvalidArgumentError` if
        the operator is singular.
    """
    ...
  
  def assert_positive_definite(self, name=...): # -> object | Operation | Any | None:
    """Returns an `Op` that asserts this operator is positive definite.

    Here, positive definite means that the quadratic form `x^H A x` has positive
    real part for all nonzero `x`.  Note that we do not require the operator to
    be self-adjoint to be positive definite.

    Args:
      name:  A name to give this `Op`.

    Returns:
      An `Assert` `Op`, that, when run, will raise an `InvalidArgumentError` if
        the operator is not positive definite.
    """
    ...
  
  def assert_self_adjoint(self, name=...): # -> object | _dispatcher_for_no_op | Operation | None:
    """Returns an `Op` that asserts this operator is self-adjoint.

    Here we check that this operator is *exactly* equal to its hermitian
    transpose.

    Args:
      name:  A string name to prepend to created ops.

    Returns:
      An `Assert` `Op`, that, when run, will raise an `InvalidArgumentError` if
        the operator is not self-adjoint.
    """
    ...
  
  def matmul(self, x, adjoint=..., adjoint_arg=..., name=...): # -> LinearOperator:
    """Transform [batch] matrix `x` with left multiplication:  `x --> Ax`.

    ```python
    # Make an operator acting like batch matrix A.  Assume A.shape = [..., M, N]
    operator = LinearOperator(...)
    operator.shape = [..., M, N]

    X = ... # shape [..., N, R], batch matrix, R > 0.

    Y = operator.matmul(X)
    Y.shape
    ==> [..., M, R]

    Y[..., :, r] = sum_j A[..., :, j] X[j, r]
    ```

    Args:
      x: `LinearOperator` or `Tensor` with compatible shape and same `dtype` as
        `self`. See class docstring for definition of compatibility.
      adjoint: Python `bool`.  If `True`, left multiply by the adjoint: `A^H x`.
      adjoint_arg:  Python `bool`.  If `True`, compute `A x^H` where `x^H` is
        the hermitian transpose (transposition and complex conjugation).
      name:  A name for this `Op`.

    Returns:
      A `LinearOperator` or `Tensor` with shape `[..., M, R]` and same `dtype`
        as `self`.
    """
    ...
  
  def __matmul__(self, other): # -> LinearOperator:
    ...
  
  def matvec(self, x, adjoint=..., name=...): # -> Any:
    """Transform [batch] vector `x` with left multiplication:  `x --> Ax`.

    ```python
    # Make an operator acting like batch matrix A.  Assume A.shape = [..., M, N]
    operator = LinearOperator(...)

    X = ... # shape [..., N], batch vector

    Y = operator.matvec(X)
    Y.shape
    ==> [..., M]

    Y[..., :] = sum_j A[..., :, j] X[..., j]
    ```

    Args:
      x: `Tensor` with compatible shape and same `dtype` as `self`.
        `x` is treated as a [batch] vector meaning for every set of leading
        dimensions, the last dimension defines a vector.
        See class docstring for definition of compatibility.
      adjoint: Python `bool`.  If `True`, left multiply by the adjoint: `A^H x`.
      name:  A name for this `Op`.

    Returns:
      A `Tensor` with shape `[..., M]` and same `dtype` as `self`.
    """
    ...
  
  def determinant(self, name=...): # -> Any:
    """Determinant for every batch member.

    Args:
      name:  A name for this `Op`.

    Returns:
      `Tensor` with shape `self.batch_shape` and same `dtype` as `self`.

    Raises:
      NotImplementedError:  If `self.is_square` is `False`.
    """
    ...
  
  def log_abs_determinant(self, name=...):
    """Log absolute value of determinant for every batch member.

    Args:
      name:  A name for this `Op`.

    Returns:
      `Tensor` with shape `self.batch_shape` and same `dtype` as `self`.

    Raises:
      NotImplementedError:  If `self.is_square` is `False`.
    """
    ...
  
  def solve(self, rhs, adjoint=..., adjoint_arg=..., name=...): # -> LinearOperator | Any:
    """Solve (exact or approx) `R` (batch) systems of equations: `A X = rhs`.

    The returned `Tensor` will be close to an exact solution if `A` is well
    conditioned. Otherwise closeness will vary. See class docstring for details.

    Examples:

    ```python
    # Make an operator acting like batch matrix A.  Assume A.shape = [..., M, N]
    operator = LinearOperator(...)
    operator.shape = [..., M, N]

    # Solve R > 0 linear systems for every member of the batch.
    RHS = ... # shape [..., M, R]

    X = operator.solve(RHS)
    # X[..., :, r] is the solution to the r'th linear system
    # sum_j A[..., :, j] X[..., j, r] = RHS[..., :, r]

    operator.matmul(X)
    ==> RHS
    ```

    Args:
      rhs: `Tensor` with same `dtype` as this operator and compatible shape.
        `rhs` is treated like a [batch] matrix meaning for every set of leading
        dimensions, the last two dimensions defines a matrix.
        See class docstring for definition of compatibility.
      adjoint: Python `bool`.  If `True`, solve the system involving the adjoint
        of this `LinearOperator`:  `A^H X = rhs`.
      adjoint_arg:  Python `bool`.  If `True`, solve `A X = rhs^H` where `rhs^H`
        is the hermitian transpose (transposition and complex conjugation).
      name:  A name scope to use for ops added by this method.

    Returns:
      `Tensor` with shape `[...,N, R]` and same `dtype` as `rhs`.

    Raises:
      NotImplementedError:  If `self.is_non_singular` or `is_square` is False.
    """
    ...
  
  def solvevec(self, rhs, adjoint=..., name=...): # -> Any:
    """Solve single equation with best effort: `A X = rhs`.

    The returned `Tensor` will be close to an exact solution if `A` is well
    conditioned. Otherwise closeness will vary. See class docstring for details.

    Examples:

    ```python
    # Make an operator acting like batch matrix A.  Assume A.shape = [..., M, N]
    operator = LinearOperator(...)
    operator.shape = [..., M, N]

    # Solve one linear system for every member of the batch.
    RHS = ... # shape [..., M]

    X = operator.solvevec(RHS)
    # X is the solution to the linear system
    # sum_j A[..., :, j] X[..., j] = RHS[..., :]

    operator.matvec(X)
    ==> RHS
    ```

    Args:
      rhs: `Tensor` with same `dtype` as this operator.
        `rhs` is treated like a [batch] vector meaning for every set of leading
        dimensions, the last dimension defines a vector.  See class docstring
        for definition of compatibility regarding batch dimensions.
      adjoint: Python `bool`.  If `True`, solve the system involving the adjoint
        of this `LinearOperator`:  `A^H X = rhs`.
      name:  A name scope to use for ops added by this method.

    Returns:
      `Tensor` with shape `[...,N]` and same `dtype` as `rhs`.

    Raises:
      NotImplementedError:  If `self.is_non_singular` or `is_square` is False.
    """
    ...
  
  def adjoint(self, name: str = ...) -> LinearOperator:
    """Returns the adjoint of the current `LinearOperator`.

    Given `A` representing this `LinearOperator`, return `A*`.
    Note that calling `self.adjoint()` and `self.H` are equivalent.

    Args:
      name:  A name for this `Op`.

    Returns:
      `LinearOperator` which represents the adjoint of this `LinearOperator`.
    """
    ...
  
  H = ...
  def inverse(self, name: str = ...) -> LinearOperator:
    """Returns the Inverse of this `LinearOperator`.

    Given `A` representing this `LinearOperator`, return a `LinearOperator`
    representing `A^-1`.

    Args:
      name: A name scope to use for ops added by this method.

    Returns:
      `LinearOperator` representing inverse of this matrix.

    Raises:
      ValueError: When the `LinearOperator` is not hinted to be `non_singular`.
    """
    ...
  
  def cholesky(self, name: str = ...) -> LinearOperator:
    """Returns a Cholesky factor as a `LinearOperator`.

    Given `A` representing this `LinearOperator`, if `A` is positive definite
    self-adjoint, return `L`, where `A = L L^T`, i.e. the cholesky
    decomposition.

    Args:
      name:  A name for this `Op`.

    Returns:
      `LinearOperator` which represents the lower triangular matrix
      in the Cholesky decomposition.

    Raises:
      ValueError: When the `LinearOperator` is not hinted to be positive
        definite and self adjoint.
    """
    ...
  
  def to_dense(self, name=...): # -> LinearOperator:
    """Return a dense (batch) matrix representing this operator."""
    ...
  
  def diag_part(self, name=...): # -> Any:
    """Efficiently get the [batch] diagonal part of this operator.

    If this operator has shape `[B1,...,Bb, M, N]`, this returns a
    `Tensor` `diagonal`, of shape `[B1,...,Bb, min(M, N)]`, where
    `diagonal[b1,...,bb, i] = self.to_dense()[b1,...,bb, i, i]`.

    ```
    my_operator = LinearOperatorDiag([1., 2.])

    # Efficiently get the diagonal
    my_operator.diag_part()
    ==> [1., 2.]

    # Equivalent, but inefficient method
    tf.linalg.diag_part(my_operator.to_dense())
    ==> [1., 2.]
    ```

    Args:
      name:  A name for this `Op`.

    Returns:
      diag_part:  A `Tensor` of same `dtype` as self.
    """
    ...
  
  def trace(self, name=...):
    """Trace of the linear operator, equal to sum of `self.diag_part()`.

    If the operator is square, this is also the sum of the eigenvalues.

    Args:
      name:  A name for this `Op`.

    Returns:
      Shape `[B1,...,Bb]` `Tensor` of same `dtype` as `self`.
    """
    ...
  
  def add_to_tensor(self, x, name=...):
    """Add matrix represented by this operator to `x`.  Equivalent to `A + x`.

    Args:
      x:  `Tensor` with same `dtype` and shape broadcastable to `self.shape`.
      name:  A name to give this `Op`.

    Returns:
      A `Tensor` with broadcast shape and same `dtype` as `self`.
    """
    ...
  
  def eigvals(self, name=...):
    """Returns the eigenvalues of this linear operator.

    If the operator is marked as self-adjoint (via `is_self_adjoint`)
    this computation can be more efficient.

    Note: This currently only supports self-adjoint operators.

    Args:
      name:  A name for this `Op`.

    Returns:
      Shape `[B1,...,Bb, N]` `Tensor` of same `dtype` as `self`.
    """
    ...
  
  def cond(self, name=...):
    """Returns the condition number of this linear operator.

    Args:
      name:  A name for this `Op`.

    Returns:
      Shape `[B1,...,Bb]` `Tensor` of same `dtype` as `self`.
    """
    ...
  
  def __getitem__(self, slices): # -> Any:
    ...
  
  __composite_gradient__ = ...


class _LinearOperatorSpec(type_spec.BatchableTypeSpec):
  """A tf.TypeSpec for `LinearOperator` objects."""
  __slots__ = ...
  def __init__(self, param_specs, non_tensor_params, prefer_static_fields) -> None:
    """Initializes a new `_LinearOperatorSpec`.

    Args:
      param_specs: Python `dict` of `tf.TypeSpec` instances that describe
        kwargs to the `LinearOperator`'s constructor that are `Tensor`-like or
        `CompositeTensor` subclasses.
      non_tensor_params: Python `dict` containing non-`Tensor` and non-
        `CompositeTensor` kwargs to the `LinearOperator`'s constructor.
      prefer_static_fields: Python `tuple` of strings corresponding to the names
        of `Tensor`-like args to the `LinearOperator`s constructor that may be
        stored as static values, if known. These are typically shapes, indices,
        or axis values.
    """
    ...
  
  @classmethod
  def from_operator(cls, operator): # -> Self:
    """Builds a `_LinearOperatorSpec` from a `LinearOperator` instance.

    Args:
      operator: An instance of `LinearOperator`.

    Returns:
      linear_operator_spec: An instance of `_LinearOperatorSpec` to be used as
        the `TypeSpec` of `operator`.
    """
    ...
  


def make_composite_tensor(cls, module_name=...):
  """Class decorator to convert `LinearOperator`s to `CompositeTensor`."""
  ...

