"""
This type stub file was generated by pyright.
"""

class Flag:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    def reset(self, arg0: bool) -> None:
        ...
    
    def value(self) -> bool:
        ...
    


class Flags:
    enable_aggressive_constant_replication: Flag
    enable_colocation_key_propagation_in_while_op_lowering: Flag
    enable_function_pruning_before_inlining: Flag
    enable_nested_function_shape_inference: Flag
    enable_quantized_dtypes_training: Flag
    enable_skip_encapsulation_for_non_tpu_graphs: Flag
    enable_tf2min_ici_weight: Flag
    graph_building_optimization: Flag
    more_stack_traces: Flag
    op_building_optimization: Flag
    publish_function_graphs: Flag
    saved_model_fingerprinting: Flag
    test_only_experiment_1: Flag
    test_only_experiment_2: Flag
    tf_shape_default_int64: Flag
    def __init__(self) -> None:
        ...
    


