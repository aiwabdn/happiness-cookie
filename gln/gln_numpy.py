# %%
from typing import Sequence, Optional, Callable
from test_mnist import get_mnist_metrics

import numpy as np


def sigmoid(X):
    return 1 / (1 + np.exp(-X))


class Neuron():
    def __init__(self,
                 input_size: int,
                 context_size: int,
                 context_map_size: int = 4,
                 output_clipping: float = 0.01,
                 weight_clipping: float = 5,
                 mu: float = 0.0,
                 std: float = 0.1,
                 learning_rate: float = 0.01):
        self._projection = np.random.normal(loc=mu,
                                            scale=std,
                                            size=(context_map_size,
                                                  context_size))
        self._projection /= np.linalg.norm(self._projection,
                                           ord=2,
                                           axis=1,
                                           keepdims=True)
        self._projection_bias = np.random.normal(loc=mu,
                                                 scale=std,
                                                 size=(context_map_size, 1))
        self._weights = np.ones(shape=(2**context_map_size,
                                       input_size)) * (1 / input_size)
        self._boolean_converter = np.array([[2**i]
                                            for i in range(context_map_size)])
        self._output_clipping = output_clipping
        self._weight_clipping = weight_clipping
        self.learning_rate = learning_rate

    def set_learning_rate(self, lr):
        self.learning_rate = lr

    def predict(self, logits, context_input, targets=None):
        projected = self._projection.dot(context_input)
        if projected.ndim == 1:
            projected = projected.reshape(-1, 1)

        mapped_context_binary = (projected > self._projection_bias).astype(
            np.int)
        current_context_indices = np.sum(mapped_context_binary *
                                         self._boolean_converter,
                                         axis=0)
        current_selected_weights = self._weights[current_context_indices, :]

        output_logits = current_selected_weights.dot(logits)
        if output_logits.ndim > 1:
            output_logits = output_logits.diagonal()

        if targets is not None:
            sigmoids = np.clip(sigmoid(output_logits), self._output_clipping,
                               1 - self._output_clipping)
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
                 output_clipping: float = 0.01,
                 weight_clipping: float = 5,
                 bias: bool = True,
                 mu: float = 0.0,
                 std: float = 0.1):

        if size == 1:
            bias = False

        if bias:
            self._neurons = [
                Neuron(input_size, context_size, context_map_size,
                       output_clipping, weight_clipping, mu, std,
                       learning_rate) for _ in range(max(1, size - 1))
            ]
            self._bias = np.random.uniform(output_clipping,
                                           1 - output_clipping)
        else:
            self._neurons = [
                Neuron(input_size, context_size, context_map_size,
                       output_clipping, weight_clipping, mu, std,
                       learning_rate) for _ in range(size)
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
                 output_clipping: float = 0.01,
                 weight_clipping: float = 5,
                 bias: bool = True,
                 mu: float = 0.0,
                 std: float = 0.1):

        self.learning_rate = learning_rate
        if size == 1:
            bias = False

        if bias:
            self._bias = np.random.uniform(output_clipping,
                                           1 - output_clipping)
        else:
            self._bias = None

        self._projection = np.random.normal(loc=mu,
                                            scale=std,
                                            size=(size, context_map_size,
                                                  context_size))
        self._projection /= np.linalg.norm(self._projection,
                                           axis=2,
                                           keepdims=True)
        self._projection_bias = np.random.normal(loc=mu,
                                                 scale=std,
                                                 size=(size, context_map_size,
                                                       1))
        self._boolean_converter = np.array([[2**i]
                                            for i in range(context_map_size)])
        self._weights = np.ones(shape=(size, 2**context_map_size,
                                       input_size)) * (1 / input_size)
        self._output_clipping = output_clipping
        self._weight_clipping = weight_clipping
        self.size = size

    def set_learning_rate(self, lr):
        self.learning_rate = lr

    def predict(self, logits, context_inputs, targets=None):
        projected = np.matmul(self._projection, context_inputs)
        mapped_context_binary = (projected > self._projection_bias).astype(
            np.int)
        current_context_indices = np.sum(mapped_context_binary *
                                         self._boolean_converter,
                                         axis=1)
        current_selected_weights = self._weights[np.arange(self.size).reshape(
            -1, 1), current_context_indices, :]

        output_logits = np.matmul(current_selected_weights,
                                  logits).diagonal(axis1=1, axis2=2)

        if self._bias is not None:
            output_logits.setflags(write=1)
            output_logits[0] = self._bias

        if targets is not None:
            sigmoids = np.clip(sigmoid(output_logits), self._output_clipping,
                               1 - self._output_clipping)
            update_value = self.learning_rate * np.expand_dims(
                (sigmoids - targets), axis=1) * logits
            self._weights[np.arange(self.size).reshape(
                -1, 1), current_context_indices, :] = np.clip(
                    self._weights[np.arange(self.size).
                                  reshape(-1, 1), current_context_indices, :] -
                    np.transpose(update_value, (0, 2, 1)),
                    -self._weight_clipping, self._weight_clipping)

        return np.squeeze(output_logits)


class GLN():
    def __init__(self,
                 layer_sizes: Sequence[int],
                 input_size: int,
                 context_size: int,
                 base_predictor: Optional[
                     Callable[[np.ndarray], np.ndarray]] = None,
                 context_map_size: int = 4,
                 layer_bias: bool = True,
                 learning_rate: float = 1e-2,
                 output_clipping: float = 0.01,
                 weight_clipping: float = 5.0):

        self.base_predictor = base_predictor
        self.layers = []
        for idx, size in enumerate(layer_sizes):
            if idx == 0:
                layer = Linear(size, input_size, context_size,
                               context_map_size, learning_rate,
                               output_clipping, weight_clipping, layer_bias)
            else:
                layer = Linear(size, layer_sizes[idx - 1], context_size,
                               context_map_size, learning_rate,
                               output_clipping, weight_clipping, layer_bias)
            self.layers.append(layer)
        self.set_learning_rate(learning_rate)

    def set_learning_rate(self, lr):
        for l in self.layers:
            l.set_learning_rate(lr)

    def predict(self, inputs, context_inputs, targets=None):
        if callable(self.base_predictor):
            out = self.base_predictor(inputs)
        else:
            out = inputs
        for l in self.layers:
            out = l.predict(out, context_inputs, targets)

        return sigmoid(out)


# %%
if __name__ == '__main__':
    m = GLN(layer_sizes=[4, 4, 1], input_size=784, context_size=784)
    acc, conf_mat, prfs = get_mnist_metrics(m, batch_size=8, mnist_class=3)
    print('Accuracy:', acc)
    print('Confusion matrix:\n', conf_mat)
    print('Prec-Rec-F:\n', prfs)
