from typing import Callable, Optional, Sequence


def GLN(backend: str,
        layer_sizes: Sequence[int],
        input_size: int,
        context_map_size: int = 4,
        num_classes: Optional[int] = None,
        base_predictor: Optional[Callable] = None,
        learning_rate: float = 1e-2,
        pred_clipping: float = 1e-3,
        weight_clipping: float = 5.0,
        bias: bool = True,
        context_bias: bool = True):

    if backend == 'jax':
        from pygln.jax import GLN
        return GLN(layer_sizes, input_size, context_map_size, num_classes,
                   base_predictor, learning_rate, pred_clipping,
                   weight_clipping, bias, context_bias)

    elif backend == 'numpy':
        from pygln.numpy import GLN
        return GLN(layer_sizes, input_size, context_map_size, num_classes,
                   base_predictor, learning_rate, pred_clipping,
                   weight_clipping, bias, context_bias)

    elif backend == 'pytorch':
        from pygln.pytorch import GLN
        return GLN(layer_sizes, input_size, context_map_size, num_classes,
                   base_predictor, learning_rate, pred_clipping,
                   weight_clipping, bias, context_bias)

    elif backend == 'tf':
        from pygln.tf import GLN
        return GLN(layer_sizes, input_size, context_map_size, num_classes,
                   base_predictor, learning_rate, pred_clipping,
                   weight_clipping, bias, context_bias)

    else:
        raise NotImplementedError(f"No implementation for backend {backend}.")
