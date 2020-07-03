# %%
import numpy as np
import torch
from torch import nn
from typing import Callable, Optional, Sequence

from test_mnist import get_mnist_metrics

if torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'


def data_transform(X, y):
    return torch.as_tensor(X, dtype=torch.float64).to(DEVICE), torch.as_tensor(
        y, dtype=torch.float64).to(DEVICE)


def result_transform(output):
    return output.detach().cpu().numpy()


class Neuron(nn.Module):
    def __init__(self,
                 input_size: int,
                 context_size: int,
                 context_map_size: int = 4,
                 output_clipping: float = 0.01,
                 weight_clipping: float = 5,
                 mu: float = 0.0,
                 std: float = 0.1,
                 learning_rate: float = 0.01):
        super(Neuron, self).__init__()
        # context function for halfspace gating
        self._projection = nn.Parameter(torch.as_tensor(
            np.random.normal(loc=mu,
                             scale=std,
                             size=(context_map_size, context_size))),
                                        requires_grad=False)
        # scale by norm
        self._projection /= torch.norm(self._projection, dim=1, keepdim=True)
        # constant values for halfspace gating
        self._projection_bias = nn.Parameter(torch.normal(
            mean=mu, std=std, size=(context_map_size, 1)),
                                             requires_grad=False)
        # weights for the neuron
        self._weights = nn.Parameter(
            torch.ones(size=(2**context_map_size, input_size),
                       dtype=torch.float64) * (1 / input_size),
            requires_grad=False)
        # array to convert mapped_context_binary context to index
        self._boolean_converter = nn.Parameter(torch.as_tensor(
            np.array([[2**i] for i in range(context_map_size)])).float(),
                                               requires_grad=False)
        # clip values
        self._output_clipping = output_clipping
        self._weight_clipping = weight_clipping
        self.learning_rate = learning_rate

    def set_learning_rate(self, lr):
        self.learning_rate = lr

    def predict(self, logits, context_inputs, targets=None):
        # project side information and determine context index
        projected = torch.matmul(self._projection, context_inputs)
        if projected.ndim == 1:
            projected = projected.reshape(-1, 1)

        mapped_context_binary = (projected > self._projection_bias).int()
        current_context_indices = torch.squeeze(
            torch.sum(mapped_context_binary * self._boolean_converter,
                      dim=0)).long()

        # select weights for current batch
        current_selected_weights = self._weights[current_context_indices, :]
        # compute logit output
        output_logits = torch.matmul(current_selected_weights,
                                     logits).diagonal()
        if output_logits.ndim > 1:
            output_logits = output_logits.diagonal()

        if targets is not None:
            # compute output and clip
            sigmoids = torch.clamp(torch.sigmoid(output_logits),
                                   self._output_clipping,
                                   1 - self._output_clipping)
            # compute update
            update_value = self.learning_rate * (sigmoids - targets) * logits
            # iterate through selected contexts and update
            for idx, ci in enumerate(current_context_indices):
                self._weights[ci, :] = torch.clamp(
                    self._weights[ci, :] - update_value[:, idx],
                    -self._weight_clipping, self._weight_clipping)

        return output_logits

    def extra_repr(self):
        return f'input_size={self._weights.size(1)}, context_map_size={self._projection.size(0)}'


class CustomLinear(nn.Module):
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

        super(CustomLinear, self).__init__()
        if size == 1:
            bias = False

        if bias:
            self._neurons = nn.ModuleList([
                Neuron(input_size, context_size, context_map_size,
                       output_clipping, weight_clipping, mu, std,
                       learning_rate) for _ in range(max(1, size - 1))
            ])
            self._bias = np.random.uniform(output_clipping,
                                           1 - output_clipping)
        else:
            self._neurons = nn.ModuleList([
                Neuron(input_size, context_size, context_map_size,
                       output_clipping, weight_clipping, mu, std,
                       learning_rate) for _ in range(size)
            ])
            self._bias = None

    def set_learning_rate(self, lr):
        for n in self._neurons:
            n.set_learning_rate(lr)

    def predict(self, logits, context_inputs, targets=None):
        output_logits = []
        if self._bias:
            output_logits.append(
                (torch.ones(logits.size(-1), dtype=torch.float64) *
                 self._bias).to(DEVICE))

        # collect outputs from all neurons
        for n in self._neurons:
            output_logits.append(n.predict(logits, context_inputs, targets))

        output = torch.squeeze(torch.stack(output_logits))
        return output

    def extra_repr(self):
        return f'bias={self._bias}'


class Linear(nn.Module):
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
        super(Linear, self).__init__()

        self.learning_rate = learning_rate
        if size == 1:
            bias = False

        if bias:
            self._bias = np.random.uniform(output_clipping,
                                           1 - output_clipping)
        else:
            self._bias = None

        # context function for halfspace gating
        self._projection = nn.Parameter(torch.as_tensor(
            np.random.normal(loc=mu,
                             scale=std,
                             size=(size, context_map_size, context_size))),
                                        requires_grad=False)
        # scale by norm
        self._projection /= torch.norm(self._projection, dim=2, keepdim=True)
        # constant values for halfspace gating
        self._projection_bias = nn.Parameter(torch.as_tensor(
            np.random.normal(loc=mu,
                             scale=std,
                             size=(size, context_map_size, 1))),
                                             requires_grad=False)
        # array to convert mapped_context_binary context to index
        self._boolean_converter = nn.Parameter(torch.as_tensor(
            np.array([[2**i] for i in range(context_map_size)])),
                                               requires_grad=False)
        # weights for the whole layer
        self._weights = nn.Parameter(
            torch.ones(size=(size, 2**context_map_size, input_size),
                       dtype=torch.float64) * (1 / input_size),
            requires_grad=False)
        # clipping value for outputs of neurons
        self._output_clipping = output_clipping
        # clipping value for weights of layer
        self._weight_clipping = weight_clipping
        self.size = size

    def set_learning_rate(self, lr):
        self.learning_rate = lr

    def predict(self, logits, context_inputs, targets=None):
        # project side information and determine context index
        projected = torch.matmul(self._projection, context_inputs)
        mapped_context_binary = (projected > self._projection_bias).int()
        current_context_indices = torch.sum(mapped_context_binary *
                                            self._boolean_converter,
                                            dim=1)

        # select all context across all neurons in layer
        current_selected_weights = self._weights[torch.arange(
            self.size).reshape(-1, 1), current_context_indices, :]

        # compute logit output
        # matmul duplicates results, so take diagonal
        output_logits = torch.matmul(current_selected_weights,
                                     logits).diagonal(dim1=1, dim2=2)

        # if not final layer
        if self._bias is not None:
            # assign output of first neuron to bias
            # done for ease of computation
            output_logits[0] = self._bias

        if targets is not None:
            # compute sigmoid of output and clip
            sigmoids = torch.clamp(torch.sigmoid(output_logits),
                                   self._output_clipping,
                                   1 - self._output_clipping)
            # compute update
            update_values = self.learning_rate * torch.unsqueeze(
                (sigmoids - targets), dim=1) * logits
            # update selected weights and clip
            self._weights[torch.arange(self.size).reshape(
                -1, 1), current_context_indices, :] = torch.clamp(
                    self._weights[torch.arange(self.size).
                                  reshape(-1, 1), current_context_indices, :] -
                    update_values.permute(0, 2, 1), -self._weight_clipping,
                    self._weight_clipping)

        return torch.squeeze(output_logits)

    def extra_repr(self):
        return 'input_size={}, neurons={}, context_map_size={}, bias={}'.format(
            self._weights.size(2), self._projection.size(0),
            self._projection.size(1), self._bias)


class GLN(nn.Module):
    def __init__(self,
                 layer_sizes: Sequence[int],
                 input_size: int,
                 context_size: int,
                 base_predictor: Optional[
                     Callable[[torch.Tensor], torch.Tensor]] = None,
                 context_map_size: int = 4,
                 layer_bias: bool = True,
                 learning_rate: float = 1e-2,
                 output_clipping: float = 0.01,
                 weight_clipping: float = 5.0):
        super(GLN, self).__init__()

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

        self.layers = nn.ModuleList(self.layers)
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

        return torch.sigmoid(out)


# %%
if __name__ == '__main__':
    m = GLN([4, 4, 1], 784, 784).to(DEVICE)
    acc, conf_mat, prfs = get_mnist_metrics(m,
                                            mnist_class=3,
                                            batch_size=8,
                                            data_transform=data_transform,
                                            result_transform=result_transform)
    print('Accuracy:', acc)
    print('Confusion matrix:\n', conf_mat)
    print('Prec-Rec-F:\n', prfs)

# %%
