import numpy as np
import torch
from scipy.special import logit as slogit
from torch import nn
from typing import Callable, Optional, Sequence, Union

from ..base import GLNBase


def data_transform(X: np.ndarray):
    return torch.Tensor(X)


def result_transform(X: torch.Tensor):
    return X.cpu().numpy()


class Linear(nn.Module):
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
            self.bias = torch.empty((1, 1, self.num_classes))
            self.bias.uniform_(slogit(self.pred_clipping),
                               slogit(1 - self.pred_clipping))
            self.size = size - 1
        else:
            self.bias = None
            self.size = size

        self._context_maps = torch.as_tensor(
            np.random.normal(size=(self.num_classes, self.size,
                                   context_map_size, context_size)),
            dtype=torch.float32)
        self._context_maps /= torch.norm(self._context_maps,
                                         dim=-1,
                                         keepdim=True)
        self._context_maps = nn.Parameter(self._context_maps)

        # constant values for halfspace gating
        if context_bias:
            context_bias_shape = (self.num_classes, self.size,
                                  context_map_size, 1)
            self._context_bias = nn.Parameter(
                torch.tensor(np.random.normal(size=context_bias_shape),
                             dtype=torch.float32),
                requires_grad=False)
        else:
            self._context_bias = 0.0

        # array to convert mapped_context_binary context to index
        self._boolean_converter = nn.Parameter(torch.as_tensor(
            np.array([[2**i] for i in range(context_map_size)])),
                                               requires_grad=False)

        # weights for the whole layer
        weights_shape = (self.num_classes, self.size, 2**context_map_size,
                         input_size)
        self._weights = nn.Parameter(torch.full(size=weights_shape,
                                                fill_value=1 / input_size,
                                                dtype=torch.float32),
                                     requires_grad=False)

    def set_learning_rate(self, lr):
        self.learning_rate = lr

    def predict(self, logit, context, target=None):
        # project side information and determine context index
        distances = torch.matmul(self._context_maps, context.T)
        mapped_context_binary = (distances > self._context_bias).int()
        current_context_indices = torch.sum(mapped_context_binary *
                                            self._boolean_converter,
                                            dim=-2)

        # select all context across all neurons in layer
        current_selected_weights = self._weights[
            torch.arange(self.num_classes).reshape(-1, 1, 1),
            torch.arange(self.size).reshape(1, -1, 1
                                            ), current_context_indices, :]

        if logit.ndim == 2:
            logit = torch.unsqueeze(logit, dim=-1)

        output_logits = torch.clamp(torch.matmul(
            current_selected_weights,
            torch.unsqueeze(logit.T, dim=-3)).diagonal(dim1=-2, dim2=-1),
                                    min=slogit(self.pred_clipping),
                                    max=slogit(1 - self.pred_clipping)).T

        if target is not None:
            sigmoids = torch.sigmoid(output_logits)
            # compute update
            diff = sigmoids - torch.unsqueeze(target, dim=1)
            update_values = self.learning_rate * torch.unsqueeze(
                diff, dim=-1) * torch.unsqueeze(logit.permute(0, 2, 1), dim=1)
            self._weights[
                torch.arange(self.num_classes).reshape(-1, 1, 1),
                torch.arange(self.size).
                reshape(1, -1, 1), current_context_indices, :] = torch.clamp(
                    self.
                    _weights[torch.arange(self.num_classes).reshape(-1, 1, 1),
                             torch.arange(self.size).
                             reshape(1, -1, 1), current_context_indices, :] -
                    update_values.permute(2, 1, 0, 3), -self.weight_clipping,
                    self.weight_clipping)

        if self.bias is not None:
            bias_append = torch.cat([self.bias] * output_logits.shape[0],
                                    dim=0)
            output_logits = torch.cat([bias_append, output_logits], dim=1)

        return output_logits

    def extra_repr(self):
        return 'input_size={}, neurons={}, context_map_size={}, bias={}'.format(
            self._weights.size(2), self._context_maps.size(0),
            self._context_maps.size(1), self.bias)


class GLN(nn.Module, GLNBase):
    def __init__(self,
                 layer_sizes: Sequence[int],
                 input_size: int,
                 context_map_size: int = 4,
                 classes: Optional[Union[int, Sequence[object]]] = None,
                 base_predictor: Optional[
                     Callable[[np.ndarray], np.ndarray]] = None,
                 learning_rate: float = 1e-4,
                 pred_clipping: float = 1e-3,
                 weight_clipping: float = 5.0,
                 bias: bool = True,
                 context_bias: bool = True):
        nn.Module.__init__(self)
        GLNBase.__init__(self, layer_sizes, input_size, context_map_size,
                         classes, base_predictor, learning_rate, pred_clipping,
                         weight_clipping, bias, context_bias)

        # Initialize layers
        self.layers = nn.ModuleList()
        previous_size = self.base_pred_size
        if bias:
            self.base_bias = np.random.uniform(low=slogit(pred_clipping),
                                               high=slogit(1 - pred_clipping))
        for idx, size in enumerate(self.layer_sizes):
            layer = Linear(size, previous_size, self.input_size,
                           self.context_map_size, self.num_classes,
                           self.learning_rate, self.pred_clipping,
                           self.weight_clipping, self.bias, self.context_bias)
            self.layers.append(layer)
            previous_size = size

    def set_learning_rate(self, lr: float):
        for layer in self.layers:
            layer.set_learning_rate(lr)

    def predict(self, input: torch.Tensor, target: torch.Tensor = None):
        # Base predictions
        base_preds = self.base_predictor(input)

        # Context
        context = input

        # Target
        if target is not None:
            target = nn.functional.one_hot(target.long(), self.num_classes)

        # Base logits
        base_preds = torch.clamp(base_preds,
                                 min=self.pred_clipping,
                                 max=(1.0 - self.pred_clipping))
        logits = torch.log(base_preds / (1.0 - base_preds))
        if self.bias:
            logits[:, 0] = self.base_bias

        # Layers
        for n, layer in enumerate(self.layers):
            logits = layer.predict(logit=logits,
                                   context=context,
                                   target=target)
        return torch.sigmoid(torch.squeeze(logits))
