# %%
import numpy as np
import torch
from torch import nn
from typing import Callable, Optional, Sequence

from pygln.datasets import get_mnist_metrics

if torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'


def data_transform(X: torch.Tensor, y: torch.Tensor):
    return torch.as_tensor(X, dtype=torch.float64).to(DEVICE), torch.as_tensor(
        y, dtype=torch.float64).to(DEVICE)


def result_transform(output: torch.Tensor):
    return output.detach().cpu().numpy()


def logit(x: torch.Tensor):
    return torch.log(x) - torch.log1p(-x)


class Neuron(nn.Module):
    def __init__(self,
                 input_size: int,
                 context_size: int,
                 context_map_size: int = 4,
                 pred_clipping: float = 0.01,
                 weight_clipping: float = 5,
                 learning_rate: float = 0.01):
        super(Neuron, self).__init__()

        # clip values
        self._output_clipping = torch.tensor(pred_clipping)
        self._weight_clipping = torch.tensor(weight_clipping)
        # context function for halfspace gating
        self._context_maps = nn.Parameter(torch.as_tensor(
            np.random.normal(size=(context_map_size, context_size))),
                                          requires_grad=False)
        # scale by norm
        self._context_maps /= torch.norm(self._context_maps,
                                         dim=1,
                                         keepdim=True)
        # constant values for halfspace gating
        self._context_bias = nn.Parameter(torch.randn(size=(context_map_size,
                                                            1)),
                                          requires_grad=False)
        # weights for the neuron
        self._weights = nn.Parameter(torch.full(size=(2**context_map_size,
                                                      input_size),
                                                fill_value=1 / input_size,
                                                dtype=torch.float64),
                                     requires_grad=False)
        # array to convert mapped_context_binary context to index
        self._boolean_converter = nn.Parameter(torch.as_tensor(
            np.array([[2**i] for i in range(context_map_size)])).float(),
                                               requires_grad=False)

        self.set_learning_rate(learning_rate)

    def set_learning_rate(self, lr):
        self.learning_rate = lr

    def predict(self, logits, context_inputs, targets=None):
        # project side information and determine context index
        distances = torch.matmul(self._context_maps, context_inputs)
        if distances.ndim == 1:
            distances = distances.reshape(-1, 1)

        mapped_context_binary = (distances > self._context_bias).int()
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

        # clip output
        output_logits = torch.clamp(output_logits,
                                    logit(self._output_clipping),
                                    logit(1 - self._output_clipping))

        if targets is not None:
            sigmoids = torch.sigmoid(output_logits)
            # compute update
            update_value = self.learning_rate * (sigmoids - targets) * logits
            # iterate through selected contexts and update
            for idx, ci in enumerate(current_context_indices):
                self._weights[ci, :] = torch.clamp(
                    self._weights[ci, :] - update_value[:, idx],
                    -self._weight_clipping, self._weight_clipping)

        return output_logits

    def extra_repr(self):
        return f'input_size={self._weights.size(1)}, context_map_size={self._context_maps.size(0)}'


class CustomLinear(nn.Module):
    def __init__(self,
                 size: int,
                 input_size: int,
                 context_size: int,
                 context_map_size: int = 4,
                 learning_rate: float = 0.01,
                 pred_clipping: float = 0.01,
                 weight_clipping: float = 5,
                 bias: bool = True):

        super(CustomLinear, self).__init__()
        if size == 1:
            bias = False

        if bias:
            self._neurons = nn.ModuleList([
                Neuron(input_size, context_size, context_map_size,
                       pred_clipping, weight_clipping, learning_rate)
                for _ in range(max(1, size - 1))
            ])
            self._bias = torch.empty(1).uniform_(
                logit(torch.tensor(pred_clipping)),
                logit(torch.tensor(1 - pred_clipping)))
        else:
            self._neurons = nn.ModuleList([
                Neuron(input_size, context_size, context_map_size,
                       pred_clipping, weight_clipping, learning_rate)
                for _ in range(size)
            ])
            self._bias = None

    def set_learning_rate(self, lr):
        for n in self._neurons:
            n.set_learning_rate(lr)

    def predict(self, inputs, context_inputs, targets=None):
        output_logits = []
        if self._bias:
            output_logits.append(
                (torch.ones(inputs.size(-1), dtype=torch.float64) *
                 self._bias).to(DEVICE))

        # collect outputs from all neurons
        for n in self._neurons:
            output_logits.append(n.predict(inputs, context_inputs, targets))

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
                 pred_clipping: float = 0.01,
                 weight_clipping: float = 5,
                 bias: bool = True):
        super(Linear, self).__init__()

        self.size = size
        # clipping value for outputs of neurons
        self._output_clipping = torch.tensor(pred_clipping)
        # clipping value for weights of layer
        self._weight_clipping = torch.tensor(weight_clipping)
        if size == 1:
            bias = False

        if bias:
            self._bias = torch.empty(1).uniform_(
                logit(self._output_clipping), logit(1 - self._output_clipping))
        else:
            self._bias = None

        # context function for halfspace gating
        self._context_maps = nn.Parameter(torch.as_tensor(
            np.random.normal(size=(size, context_map_size, context_size))),
                                          requires_grad=False)
        # scale by norm
        self._context_maps /= torch.norm(self._context_maps,
                                         dim=2,
                                         keepdim=True)
        # constant values for halfspace gating
        self._context_bias = nn.Parameter(torch.as_tensor(
            np.random.normal(size=(size, context_map_size, 1))),
                                          requires_grad=False)
        # array to convert mapped_context_binary context to index
        self._boolean_converter = nn.Parameter(torch.as_tensor(
            np.array([[2**i] for i in range(context_map_size)])),
                                               requires_grad=False)
        # weights for the whole layer
        self._weights = nn.Parameter(torch.full(size=(size,
                                                      2**context_map_size,
                                                      input_size),
                                                fill_value=1 / input_size,
                                                dtype=torch.float64),
                                     requires_grad=False)
        self.set_learning_rate(learning_rate)

    def set_learning_rate(self, lr):
        self.learning_rate = lr

    def predict(self, logits, context_inputs, targets=None):
        # project side information and determine context index
        distances = torch.matmul(self._context_maps, context_inputs)
        mapped_context_binary = (distances > self._context_bias).int()
        current_context_indices = torch.sum(mapped_context_binary *
                                            self._boolean_converter,
                                            dim=1)

        # select all context across all neurons in layer
        current_selected_weights = self._weights[torch.arange(
            self.size).reshape(-1, 1), current_context_indices, :]

        # compute logit output
        # matmul duplicates results, so take diagonal
        output_logits = torch.clamp(torch.matmul(current_selected_weights,
                                                 logits).diagonal(dim1=1,
                                                                  dim2=2),
                                    min=logit(self._output_clipping),
                                    max=logit(1 - self._output_clipping))

        # if not final layer
        if self._bias is not None:
            # assign output of first neuron to bias
            # done for ease of computation
            output_logits[0] = self._bias

        if targets is not None:
            sigmoids = torch.sigmoid(output_logits)
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
            self._weights.size(2), self._context_maps.size(0),
            self._context_maps.size(1), self._bias)


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
                 pred_clipping: float = 0.01,
                 weight_clipping: float = 5.0):
        super(GLN, self).__init__()

        self.base_predictor = base_predictor
        self.layers = []
        for idx, size in enumerate(layer_sizes):
            if idx == 0:
                layer = Linear(size, input_size, context_size,
                               context_map_size, learning_rate, pred_clipping,
                               weight_clipping, layer_bias)
            else:
                layer = Linear(size, layer_sizes[idx - 1], context_size,
                               context_map_size, learning_rate, pred_clipping,
                               weight_clipping, layer_bias)
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
