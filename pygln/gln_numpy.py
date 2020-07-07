# %%
from typing import Callable, Optional, Sequence, Union

import numpy as np
import scipy
from sklearn.preprocessing import label_binarize

from base import GatedLinearNetwork


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
                 learning_rate: float = 0.01,
                 pred_clipping: float = 0.01,
                 weight_clipping: float = 5,
                 bias: bool = True,
                 num_classes: int = 2):

        self.num_classes = num_classes
        self.size = size
        if size == 1:
            bias = False

        if num_classes < 2:
            raise ValueError("Number of classes need to be at least 2")
        elif num_classes == 2:
            # binary case
            self.num_classes = 1

        if bias:
            self._bias = np.random.uniform(
                low=scipy.special.logit(pred_clipping),
                high=scipy.special.logit(1 - pred_clipping),
                size=(self.num_classes, 1))
        else:
            self._bias = None

        self._context_maps = np.random.normal(size=(self.num_classes, size,
                                                    context_map_size,
                                                    context_size))
        self._context_maps /= np.linalg.norm(self._context_maps,
                                             axis=-1,
                                             keepdims=True)
        self._context_bias = np.random.normal(size=(self.num_classes, size,
                                                    context_map_size, 1))
        self._boolean_converter = np.array([[2**i]
                                            for i in range(context_map_size)])
        self._weights = np.ones(shape=(self.num_classes, size,
                                       2**context_map_size,
                                       input_size)) * (1 / input_size)

        self._output_clipping = pred_clipping
        self._weight_clipping = weight_clipping
        self.set_learning_rate(learning_rate)

    def set_learning_rate(self, lr):
        self.learning_rate = lr

    def predict(self, logits, context_inputs, targets=None):
        distances = np.matmul(self._context_maps, context_inputs)
        mapped_context_binary = (distances > self._context_bias).astype(np.int)
        current_context_indices = np.sum(mapped_context_binary *
                                         self._boolean_converter,
                                         axis=-2)
        current_selected_weights = self._weights[
            np.arange(self.num_classes).reshape(-1, 1, 1),
            np.arange(self.size).reshape(1, -1, 1), current_context_indices, :]

        logits = np.expand_dims(logits, axis=-3)
        output_logits = np.clip(
            np.matmul(current_selected_weights, logits).diagonal(axis1=-2,
                                                                 axis2=-1),
            scipy.special.logit(self._output_clipping),
            scipy.special.logit(1 - self._output_clipping))

        if targets is not None:
            sigmoids = sigmoid(output_logits)
            diff = sigmoids - np.expand_dims(targets, axis=1)
            update_value = self.learning_rate * np.expand_dims(diff,
                                                               axis=2) * logits
            self._weights[
                np.arange(self.num_classes).reshape(-1, 1, 1),
                np.arange(self.size).
                reshape(1, -1, 1), current_context_indices, :] = np.clip(
                    self.
                    _weights[np.arange(self.num_classes).reshape(-1, 1, 1),
                             np.arange(self.size).
                             reshape(1, -1, 1), current_context_indices, :] -
                    np.transpose(update_value, (0, 1, 3, 2)),
                    -self._weight_clipping, self._weight_clipping)

        if self._bias is not None:
            output_logits.setflags(write=1)
            output_logits[:, 0] = self._bias

        return output_logits


class GLN(GatedLinearNetwork):
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
            context_bias: bool = True):
        super().__init__(layer_sizes, input_size, context_map_size, classes,
                         base_predictor, learning_rate, pred_clipping,
                         weight_clipping, bias, context_bias)

        self.layers = []
        previous_size = self.base_pred_size
        for size in self.layer_sizes:
            layer = Linear(size, previous_size, self.input_size,
                           self.context_map_size, self.learning_rate,
                           self.pred_clipping, self.weight_clipping, self.bias,
                           len(self.classes))
            previous_size = size
            self.layers.append(layer)

    def set_learning_rate(self, lr):
        for layer in self.layers:
            layer.set_learning_rate(lr)

    def predict(self, input, target=None):
        if callable(self.base_predictor):
            base_pred = self.base_predictor(input)
        else:
            base_pred = input
        base_pred = np.clip(base_pred,
                            a_min=self.pred_clipping,
                            a_max=(1.0 - self.pred_clipping))
        logit = scipy.special.logit(base_pred).T

        if target is not None:
            target = label_binarize(target, classes=self.classes).T

        for layer in self.layers:
            logit = layer.predict(logit, input.T, target)

        return np.squeeze(sigmoid(logit.T))


# %%
if __name__ == '__main__':
    from utils import get_mnist_metrics
    m = GLN(layer_sizes=[4, 4, 1],
            input_size=784,
            classes=range(10),
            bias=False,
            base_predictor=lambda x: (x * (1 - 2 * 0.01)) + 0.01)
    acc, conf_mat, prfs = get_mnist_metrics(m, batch_size=2, deskewed=False)
    print('Accuracy:', acc)
    print('Confusion matrix:\n', conf_mat)
    print('Prec-Rec-F:\n', prfs)
