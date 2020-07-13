from abc import ABC, abstractmethod
import numpy as np
from typing import Callable, Optional, Sequence


class OnlineUpdateModel(ABC):
    """Base class for online-update models, shared by all backend implementations."""

    @abstractmethod
    def predict(self, input, target=None):
        """
        Predict the class for the given inputs, and optionally update the weights.

        Args:
            input: Batch of input instances.
            target: Optional target class vector, triggers online update if given.

        Returns:
            Predicted class per input instance.
        """
        raise NotImplementedError()


class GLNBase(OnlineUpdateModel):
    """Base class for Gated Linear Network implementations (https://arxiv.org/abs/1910.01526)."""

    def __init__(
        self,
        layer_sizes: Sequence[int],
        input_size: int,
        context_map_size: int = 4,
        num_classes: Optional[int] = None,
        base_predictor: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        learning_rate: float = 1e-4,
        pred_clipping: float = 1e-3,
        weight_clipping: float = 5.0,
        bias: bool = True,
        context_bias: bool = True
    ):
        """... to be done..."""
        super().__init__()

        assert len(layer_sizes) > 0 and layer_sizes[-1] == 1
        self.layer_sizes = tuple(layer_sizes)

        assert input_size > 0
        self.input_size = input_size

        assert context_map_size >= 2
        self.context_map_size = context_map_size

        if num_classes is None:
            self.num_classes = 1
        else:
            assert num_classes >= 2
            self.num_classes = num_classes

        if base_predictor is None:
            self.base_predictor = (lambda x: x)
            self.base_pred_size = self.input_size
        else:
            self.base_predictor = base_predictor
            dummy_input = np.zeros(shape=(1, input_size))
            dummy_pred = self.base_predictor(dummy_input)
            assert dummy_pred.dtype in (np.float32, np.float64)
            assert dummy_pred.ndim == 2 and dummy_pred.shape[0] == 1
            self.base_pred_size = dummy_pred.shape[1]

        assert learning_rate > 0.0
        self.learning_rate = learning_rate

        assert 0.0 < pred_clipping < 1.0
        self.pred_clipping = pred_clipping

        assert weight_clipping >= 1.0
        self.weight_clipping = weight_clipping

        self.bias = bias
        self.context_bias = context_bias
