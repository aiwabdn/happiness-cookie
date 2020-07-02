# %%
from typing import Sequence
import numpy as np


class OnlineUpdateModule(object):
    def __init__(self, learning_rate: float, pred_clipping: float,
                 weight_clipping: float):
        self.learning_rate = learning_rate
        self.weight_clipping = weight_clipping

    def predict(self, preds, input, target=None):
        raise NotImplementedError()


def sigmoid(X):
    return 1 / (1 + np.exp(-X))


class Neuron(OnlineUpdateModule):
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

    def predict(self, logits, context_input, targets=None):
        projected = self._projection.dot(context_input)
        if projected.ndim == 1:
            projected = projected.reshape(-1, 1)

        mapped_context_binary = (projected > self._projection_bias).astype(
            np.int)
        current_context_indices = np.squeeze(
            np.sum(mapped_context_binary * self._boolean_converter, axis=0))
        print(current_context_indices)
        current_selected_weights = self._weights[current_context_indices, :]

        output_logits = current_selected_weights.dot(logits)
        if output_logits.ndim > 1:
            output_logits = output_logits.diagonal()

        if targets is not None:
            sigmoids = np.clip(sigmoid(output_logits), self._output_clipping,
                               1 - self._output_clipping)
            update_value = self.learning_rate * (sigmoids - targets) * logits

            for idx, ci in enumerate(current_context_indices):
                self._weights[ci, :] -= update_value[:, idx]

        return output_logits


class CustomLinear(OnlineUpdateModule):
    def __init__(self,
                 size: int,
                 input_size: int,
                 context_size: int,
                 context_map_size: int,
                 learning_rate: float,
                 output_clipping: float = 0.01,
                 weight_clipping: float = 5,
                 bias: bool = True,
                 mu: float = 0.0,
                 std: float = 0.1):
        super().__init__(learning_rate, weight_clipping)

        if size == 1:
            bias = False

        if bias:
            self._neurons = [
                Neuron(input_size, context_size, context_map_size,
                       output_clipping, weight_clipping, mu, std)
                for _ in range(max(1, size - 1))
            ]
            self._bias = np.random.uniform(output_clipping,
                                           1 - output_clipping)
        else:
            self._neurons = [
                Neuron(input_size, context_size, context_map_size,
                       output_clipping, weight_clipping, mu, std)
                for _ in range(size)
            ]
            self._bias = None

    def predict(self, logits, context_input, targets=None):
        output_logits = []

        if self._bias:
            output_logits.append(np.repeat(self._bias, logits.shape[-1]))

        for n in self._neurons:
            output_logits.append(n.predict(logits, context_input, targets))

        output = np.vstack(output_logits)
        return output


class Linear(OnlineUpdateModule):
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
        # super(Linear, self).__init__(learning_rate, output_clipping, weight_clipping)

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
        self._boolean_converter = np.array(
            [[2**i for i in range(context_map_size)]])
        self._weights = np.ones(shape=(size, 2**context_map_size,
                                       input_size)) * (1 / input_size)
        self._output_clipping = output_clipping
        self._weight_clipping = weight_clipping
        self.size = size

    def predict(self, logits, context_inputs, targets=None):
        projected = np.matmul(self._projection, context_inputs)
        mapped_context_binary = (projected > self._projection_bias).astype(
            np.int)
        current_context_indices = np.squeeze(
            self._boolean_converter.dot(mapped_context_binary))
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

        return output_logits


def GLN(OnlineUpdateModule):
    def __init__(self,
                 layer_sizes: Sequence[int],
                 input_size: int,
                 context_size: int,
                 context_map_size: int = 4,
                 learning_rate: float = 1e-4,
                 output_clipping: float = 0.05,
                 weight_clipping: float = 5.0,
                 classes: int = 2,
                 base_preds_size: int = None):
        super(GLN, self).__init__(learning_rate, output_clipping,
                                  weight_clipping)

        self.layers = []
        for idx, size in enumerate(layer_sizes):
            if idx == 0:
                layer = CustomLinear(size, base_preds_size, context_size,
                                     context_map_size, learning_rate,
                                     output_clipping, weight_clipping)
            else:
                layer = CustomLinear(size, layer_sizes[idx - 1], context_size,
                                     context_map_size, learning_rate,
                                     output_clipping, weight_clipping)
            self.layers.append(layer)

    def set_learning_rate(self, lr):
        self.learning_rate = lr

    def predict(self, base_predictions, context_input, targets=None):
        out = base_predictions
        for l in self.layers:
            out = l.predict(out, context_input, targets)

        return sigmoid(out)


# %%
n = Neuron(4, 784, 4)
c = np.random.normal(size=(784, 1))
i = np.random.normal(size=(4, 1))
t = np.array([0])
n.predict(i, c, t)
# %%
from test_mnist import get_mnist_metrics
if __name__ == '__main__':
    m = GLN(layer_sizes=[4, 4, 1], input_size=784
    acc, conf_mat, prfs = get_mnist_metrics(m, batch_size=8, mnist_class=3)
    print('Accuracy:', acc)
    print('Confusion matrix:\n', conf_mat)
    print('Prec-Rec-F:\n', prfs)

