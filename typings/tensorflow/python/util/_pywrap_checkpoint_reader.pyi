"""
This type stub file was generated by pyright.
"""

from typing import Any

class CheckpointReader:
    def __init__(self, arg0: str) -> None:
        ...
    
    @classmethod
    def CheckpointReader_GetTensor(cls, arg0: CheckpointReader, arg1: str) -> object:
        ...
    
    def debug_string(self) -> bytes:
        ...
    
    def get_variable_to_shape_map(self, *args, **kwargs) -> Any:
        ...
    


