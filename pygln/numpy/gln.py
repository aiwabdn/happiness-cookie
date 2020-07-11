import numpy as np
import scipy
from typing import Callable, Optional, Sequence, Union
from sklearn.preprocessing import label_binarize

from ..base import GLNBase


def sigmoid(X: np.ndarray):
    return 1 / (1 + np.exp(-X))


class Neuron():
    def __init__(self,
                 input_size: int,
                 context_size: int,
                 context_map_size: int = 4,
                 pred_clipping: float = 0.01,
                 weight_clipping: float = 5,
                 learning_rate: float = 0.01):
        self._context_maps = np.random.normal(size=(context_map_size,
                                                    context_size))
        self._context_maps /= np.linalg.norm(self._context_maps,
                                             ord=2,
                                             axis=1,
                                             keepdims=True)
        self._context_bias = np.random.normal(size=(context_map_size, 1))
        self._weights = np.ones(shape=(2**context_map_size,
                                       input_size)) * (1 / input_size)
        self._boolean_converter = np.array([[2**i]
                                            for i in range(context_map_size)])
        self._output_clipping = pred_clipping
        self._weight_clipping = weight_clipping
        self.set_learning_rate(learning_rate)

    def set_learning_rate(self, lr):
        self.learning_rate = lr

    def predict(self, logits, context_input, targets=None):
        distances = self._context_maps.dot(context_input)
        if distances.ndim == 1:
            distances = distances.reshape(-1, 1)

        mapped_context_binary = (distances > self._context_bias).astype(np.int)
        current_context_indices = np.sum(mapped_context_binary *
                                         self._boolean_converter,
                                         axis=0)
        current_selected_weights = self._weights[current_context_indices, :]

        output_logits = current_selected_weights.dot(logits)
        if output_logits.ndim > 1:
            output_logits = output_logits.diagonal()

        output_logits = np.clip(output_logits,
                                scipy.special.logit(self._output_clipping),
                                scipy.special.logit(1 - self._output_clipping))

        if targets is not None:
            sigmoids = sigmoid(output_logits)
            update_value = self.learning_rate * (sigmoids - targets) * logits

            for idx, ci in enumerate(current_context_indices):
                self._weights[ci, :] = np.clip(
                    self._weights[ci, :] - update_value[:, idx],
                    -self._weight_clipping, self._weight_clipping)

        return output_logits


class CustomLinear():
    def __init__(self,
                 size: int,
                 input_size: int,
                 context_size: int,
                 context_map_size: int = 4,
                 learning_rate: float = 0.01,
                 pred_clipping: float = 0.01,
                 weight_clipping: float = 5,
                 bias: bool = True):

        if size == 1:
            bias = False

        if bias:
            self._neurons = [
                Neuron(input_size, context_size, context_map_size,
                       pred_clipping, weight_clipping, learning_rate)
                for _ in range(max(1, size - 1))
            ]
            self._bias = np.random.uniform(
                scipy.special.logit(pred_clipping),
                scipy.special.logit(1 - pred_clipping))
        else:
            self._neurons = [
                Neuron(input_size, context_size, context_map_size,
                       pred_clipping, weight_clipping, learning_rate)
                for _ in range(size)
            ]
            self._bias = None

    def set_learning_rate(self, lr):
        for n in self._neurons:
            n.set_learning_rate(lr)

    def predict(self, logits, context_input, targets=None):
        output_logits = []

        if self._bias:
            output_logits.append(np.repeat(self._bias, logits.shape[-1]))

        for n in self._neurons:
            output_logits.append(n.predict(logits, context_input, targets))

        output = np.squeeze(np.vstack(output_logits))
        return output


class Linear():
    def __init__(self,
                 size: int,
                 input_size: int,
                 context_size: int,
                 context_map_size: int = 4,
                 num_classes: int = 2,
                 learning_rate: float = 0.01,
                 pred_clipping: float = 0.001,
                 weight_clipping: float = 5.0,
                 bias: bool = True,
                 context_bias: bool = True):
        super().__init__()

        assert size > 0 and input_size > 0 and context_size > 0
        assert context_map_size >= 2
        assert num_classes >= 2
        assert learning_rate > 0.0
        assert 0.0 < pred_clipping < 1.0
        assert weight_clipping >= 1.0

        self.num_classes = num_classes if num_classes > 2 else 1
        self.learning_rate = learning_rate
        # clipping value for predictions
        self.pred_clipping = pred_clipping
        # clipping value for weights of layer
        self.weight_clipping = weight_clipping

        if bias and size > 1:
            self.bias = np.random.uniform(
                low=scipy.special.logit(self.pred_clipping),
                high=scipy.special.logit(1 - self.pred_clipping),
                size=(1, 1, self.num_classes))
            self.size = size - 1
        else:
            self.bias = None
            self.size = size

        self._context_maps = np.random.normal(size=(self.num_classes,
                                                    self.size,
                                                    context_map_size,
                                                    context_size))
        self._context_maps /= np.linalg.norm(self._context_maps,
                                             axis=-1,
                                             keepdims=True)
        if context_bias:
            self._context_bias = np.random.normal(size=(self.num_classes,
                                                        self.size,
                                                        context_map_size, 1))
        else:
            self._context_bias = 0.0
        self._boolean_converter = np.array([[2**i]
                                            for i in range(context_map_size)])
        self._weights = np.full(shape=(self.num_classes, self.size,
                                       2**context_map_size, input_size),
                                fill_value=1 / input_size)

    def set_learning_rate(self, lr):
        self.learning_rate = lr

    def predict(self, logit, context, target=None):
        distances = np.matmul(self._context_maps, context.T)
        mapped_context_binary = (distances > self._context_bias).astype(np.int)
        current_context_indices = np.sum(mapped_context_binary *
                                         self._boolean_converter,
                                         axis=-2)
        current_selected_weights = np.take_along_axis(
            self._weights,
            indices=np.expand_dims(current_context_indices, axis=-1),
            axis=2)

        if logit.ndim == 2:
            logit = np.expand_dims(logit, axis=-1)

        output_logits = np.clip(
            np.matmul(current_selected_weights,
                      np.expand_dims(logit.T, axis=-3)).diagonal(axis1=-2,
                                                                 axis2=-1),
            scipy.special.logit(self.pred_clipping),
            scipy.special.logit(1 - self.pred_clipping)).T

        if target is not None:
            sigmoids = sigmoid(output_logits)
            diff = sigmoids - np.expand_dims(target, axis=1)
            update_value = self.learning_rate * np.expand_dims(
                diff, axis=-1) * np.expand_dims(np.swapaxes(logit, -1, -2),
                                                axis=1)

            np.add.at(
                self._weights, [
                    np.arange(self.num_classes).reshape(-1, 1, 1, 1),
                    np.arange(self.size).reshape(1, -1, 1, 1),
                    np.expand_dims(current_context_indices, axis=-1)
                ], -np.expand_dims(np.transpose(update_value, [2, 1, 0, 3]),
                                   axis=-2))
            self._weights = np.clip(self._weights, -self.weight_clipping,
                                    self.weight_clipping)

        if self.bias is not None:
            output_logits = np.concatenate([
                np.vstack([self.bias] * output_logits.shape[0]), output_logits
            ],
                                           axis=1)

        return output_logits


class GLN(GLNBase):
    def __init__(self,
                 layer_sizes: Sequence[int],
                 input_size: int,
                 context_map_size: int = 4,
                 classes: Optional[Union[int, Sequence[object]]] = None,
                 base_predictor: Optional[
                     Callable[[np.ndarray], np.ndarray]] = None,
                 learning_rate: float = 1e-2,
                 pred_clipping: float = 1e-3,
                 weight_clipping: float = 5.0,
                 bias: bool = True,
                 context_bias: bool = True):
        super().__init__(layer_sizes, input_size, context_map_size, classes,
                         base_predictor, learning_rate, pred_clipping,
                         weight_clipping, bias, context_bias)

        # Initialize layers
        self.layers = list()
        previous_size = self.base_pred_size
        if bias:
            self.base_bias = np.random.uniform(
                low=scipy.special.logit(pred_clipping),
                high=scipy.special.logit(1 - pred_clipping))
        for size in self.layer_sizes:
            layer = Linear(size, previous_size, self.input_size,
                           self.context_map_size, self.num_classes,
                           self.learning_rate, self.pred_clipping,
                           self.weight_clipping, self.bias, self.context_bias)
            self.layers.append(layer)
            previous_size = size

    def set_learning_rate(self, lr: float):
        for layer in self.layers:
            layer.set_learning_rate(lr)

    def predict(self, input: np.ndarray, target: np.ndarray = None):
        # Base predictions
        base_preds = self.base_predictor(input)
        base_preds = np.asarray(base_preds, dtype=float)

        # Context
        context = np.asarray(input, dtype=float)

        # Target
        if target is not None:
            target = label_binarize(target, classes=self.classes)

        # Base logits
        base_preds = np.clip(base_preds,
                             a_min=self.pred_clipping,
                             a_max=(1.0 - self.pred_clipping))
        logits = scipy.special.logit(base_preds)
        if self.bias:
            # introduce layer bias
            logits[:, 0] = self.base_bias

        # Layers
        for layer in self.layers:
            logits = layer.predict(logit=logits,
                                   context=context,
                                   target=target)

        return sigmoid(np.squeeze(logits))
