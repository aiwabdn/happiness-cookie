from abc import ABC, abstractmethod
import numpy as np
from typing import Callable, Sequence, Union


class OnlineUpdateModel(ABC):
    """Base class for online-update models, shared by all backend implementations."""

    @abstractmethod
    def predict(self, input, target=None):
        """Predict the class for the given inputs, and optionally update the weights.

        Args:
            input: Batch of input instances.
            target: Optional target class vector, triggers online update if given.

        Returns:
            Predicted class per input instance.
        """
        raise NotImplementedError()


class GatedLinearNetwork(OnlineUpdateModel):
    """Gated Linear Network, based on https://arxiv.org/abs/1910.01526."""

    def __init__(
        self,
        layer_sizes: Sequence[int],
        input_size: int,
        context_map_size: int = 4,
        classes: Union[int, Sequence[object]] = 2,
        base_predictor: Callable[[np.ndarray], np.ndarray] = (lambda x: x),
        learning_rate: float = 1e-4,
        pred_clipping: float = 1e-2,
        weight_clipping: float = 5.0,
        bias: bool = True,
        context_bias: bool = True
    ):
        """... to be done..."""
        super().__init__()

        assert len(layer_sizes) > 0 and layer_sizes[-1] == 1
        self.layer_sizes = layer_sizes

        assert input_size > 0
        self.input_size = input_size

        assert context_map_size >= 2
        self.context_map_size = context_map_size

        if isinstance(classes, int):
            assert classes >= 2
            self.classes = list(range(classes))
        else:
            assert len(classes) >= 2
            self.classes = list(classes)

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
