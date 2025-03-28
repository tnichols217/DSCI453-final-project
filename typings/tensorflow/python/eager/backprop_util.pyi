"""
This type stub file was generated by pyright.
"""

"""Shared utilities related to backprop."""
def IsTrainable(tensor_or_dtype): # -> bool:
  """Determines whether a tensor or dtype supports infinitesimal changes."""
  ...

def FlattenNestedIndexedSlices(grad): # -> IndexedSlices:
  ...

def AggregateIndexedSlicesGradients(grads): # -> defaultdict[Any, Any] | Any | list[Any] | object | SymbolicTensor | IndexedSlices | None:
  """Aggregates gradients containing `IndexedSlices`s."""
  ...

