import numpy as np
import torch
from torch import nn
from typing import Callable, Optional, Sequence, Union

from base import GLNBase


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

    def __init__(
        self,
        size: int,
        input_size: int,
        context_size: int,
        context_map_size: int,
        classes: int,
        learning_rate: float,
        pred_clipping: float,
        weight_clipping: float,
        bias: bool,
        context_bias: bool
    ):
        super().__init__()

        assert size > 0 and input_size > 0 and context_size > 0
        assert context_map_size >= 2
        assert classes >= 1
        assert learning_rate > 0.0
        assert 0.0 < pred_clipping < 1.0
        assert weight_clipping >= 1.0

        self.size = size
        self.classes = classes
        self.learning_rate = learning_rate
        # clipping value for predictions
        self.pred_clipping = torch.tensor(pred_clipping)
        # clipping value for weights of layer
        self.weight_clipping = weight_clipping

        if bias:
            self.bias = torch.empty((1, 1, self.classes, 1))
            self.bias.uniform_(logit(self.pred_clipping), logit(1 - self.pred_clipping))
        else:
            self.bias = None

        # context function for halfspace gating
        context_maps_shape = (self.classes, self.size, context_map_size, context_size)
        self.context_maps = nn.Parameter(
            torch.tensor(np.random.normal(size=context_maps_shape), dtype=torch.float32),
            requires_grad=False
        )
        # Normalize
        self.context_maps /= torch.norm(self.context_maps, dim=-1, keepdim=True)

        # constant values for halfspace gating
        if context_bias:
            context_bias_shape = (self.classes, self.size, context_map_size, 1)
            self.context_bias = nn.Parameter(
                torch.tensor(np.random.normal(size=context_bias_shape), dtype=torch.float32),
                requires_grad=False
            )
        else:
            self.context_bias = 0.0

        # array to convert mapped_context_binary context to index
        self.boolean_converter = nn.Parameter(
            torch.as_tensor(np.array([[2**i] for i in range(context_map_size)])),
            requires_grad=False
        )

        # weights for the whole layer
        input_size = input_size + int(self.bias is not None)
        weights_shape = (self.classes, self.size, 2**context_map_size, input_size)
        self.weights = nn.Parameter(
            torch.full(size=weights_shape, fill_value=1 / input_size, dtype=torch.float32),
            requires_grad=False
        )

    def set_learning_rate(self, lr):
        self.learning_rate = lr

    def predict(self, logits, context, target=None):
        # project side information and determine context index
        distances = torch.matmul(self.context_maps, context.T)
        mapped_context_binary = (distances > self.context_bias).int()
        current_context_indices = torch.sum(mapped_context_binary *
                                            self.boolean_converter,
                                            dim=-2)

        # select all context across all neurons in layer
        current_selected_weights = self.weights[
            torch.arange(self.classes).reshape(-1, 1, 1),
            torch.arange(self.size).reshape(1, -1, 1
                                            ), current_context_indices, :]

        # compute logit output
        # matmul duplicates results, so take diagonal
        logits = torch.unsqueeze(logits, dim=-3)

        # bias = tf.tile(self.bias, multiples=(batch_size, 1, 1))
        # logits = tf.concat([logits, bias], axis=-1)
        # logits = tf.expand_dims(logits, axis=-1)

        if self.bias is not None:
            logits = torch.cat([logits, self.bias], axis=3)

        output_logits = torch.clamp(torch.matmul(current_selected_weights,
                                                 logits).diagonal(dim1=-2,
                                                                  dim2=-1),
                                    min=logit(self.pred_clipping),
                                    max=logit(1 - self.pred_clipping))

        if target is not None:
            sigmoids = torch.sigmoid(output_logits)
            # compute update
            diff = sigmoids - torch.unsqueeze(target, dim=1)
            update_values = self.learning_rate * torch.unsqueeze(
                diff, dim=2) * logits
            # update selected weights and clip
            self.weights[
                torch.arange(self.classes).reshape(-1, 1, 1),
                torch.arange(self.size).
                reshape(1, -1, 1), current_context_indices, :] = torch.clamp(
                    self.
                    weights[torch.arange(self.classes).reshape(-1, 1, 1),
                             torch.arange(self.size).
                             reshape(1, -1, 1), current_context_indices, :] -
                    update_values.permute(0, 1, 3, 2), -self.weight_clipping,
                    self.weight_clipping)

        # # if not final layer
        # if self.bias is not None:
        #     # assign output of first neuron to bias
        #     output_logits[:, 0] = self.bias

        return output_logits

    def extra_repr(self):
        return 'input_size={}, neurons={}, context_map_size={}, bias={}'.format(
            self.weights.size(2), self.context_maps.size(0),
            self.context_maps.size(1), self.bias)


class GLN(nn.Module, GLNBase):

    def __init__(
        self,
        layer_sizes: Sequence[int],
        input_size: int,
        context_map_size: int = 4,
        classes: Optional[Union[int, Sequence[object]]] = None,
        base_predictor: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        learning_rate: float = 1e-4,
        pred_clipping: float = 1e-3,
        weight_clipping: float = 5.0,
        bias: bool = True,
        context_bias: bool = True
    ):
        nn.Module.__init__(self)
        GLNBase.__init__(
            self, layer_sizes, input_size, context_map_size, classes, base_predictor,
            learning_rate, pred_clipping, weight_clipping, bias, context_bias
        )

        # Initialize layers
        self.layers = nn.ModuleList()
        previous_size = self.base_pred_size
        for idx, size in enumerate(self.layer_sizes):
            layer = Linear(
                size, previous_size, self.input_size, self.context_map_size, self.num_classes,
                self.learning_rate, self.pred_clipping, self.weight_clipping, self.bias,
                self.context_bias
            )
            self.layers.append(layer)
            previous_size = size

    def set_learning_rate(self, lr):
        for layer in self.layers:
            layer.set_learning_rate(lr)

    def predict(self, input, target=None):
        # Base predictions
        base_preds = self.base_predictor(input)
        base_preds = torch.tensor(base_preds, dtype=torch.float32)

        # Context
        context = torch.tensor(input, dtype=torch.float32)

        # Target
        if target is not None:
            if self.num_classes == 1:
                target = torch.tensor(target, dtype=torch.bool)
            elif self.classes is None:
                target = torch.tensor(target, dtype=torch.int64)
            else:
                target = torch.tensor([self.classes.index(x) for x in target], dtype=torch.int64)

        # Base logits
        base_preds = torch.clamp(base_preds, min=self.pred_clipping, max=(1.0 - self.pred_clipping))
        logits = torch.log(base_preds / (1.0 - base_preds))
        logits = logits.unsqueeze(dim=1)
        logits = logits.repeat(1, self.num_classes, 1)

        # Turn target class into one-hot
        if target is not None:
            if self.num_classes == 1:
                target = torch.where(target, 1.0, 0.0).unsqueeze(dim=1)
            else:
                target = torch.nn.functional.one_hot(target, num_classes=self.num_classes)

        # Layers
        for n, layer in enumerate(self.layers):
            logits = layer.predict(logits=logits, context=context, target=target)
        logits = logits.squeeze(axis=-1)

        # Output prediction
        if self.num_classes == 1:
            return logits.squeeze(axis=-1) > 0.0
        else:
            return logits.argmax(axis=1)


# %%
if __name__ == '__main__':
    from utils import get_mnist_metrics
    m = GLN(layer_sizes=[4, 4, 1],
            input_size=784,
            classes=range(10),
            base_predictor=lambda x: (x * (1 - 2 * 0.01)) + 0.01).to(DEVICE)
    m = m.to(DEVICE)
    acc, conf_mat, prfs = get_mnist_metrics(m,
                                            batch_size=1,
                                            data_transform=data_transform,
                                            result_transform=result_transform)
    print('Accuracy:', acc)
    print('Confusion matrix:\n', conf_mat)
    print('Prec-Rec-F:\n', prfs)
