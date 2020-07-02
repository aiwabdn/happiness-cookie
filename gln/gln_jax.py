from typing import Sequence


class OnlineUpdateModule(object):

    def __init__(self, learning_rate: float, pred_clipping: float, weight_clipping: float):
        self.learning_rate = learning_rate
        self.weight_clipping = weight_clipping

    def predict(self, preds, input, target=None):
        raise NotImplementedError()


class Linear(OnlineUpdateModule):

    def __init__(
        self, size: int, preds_size: int, input_size: int, context_dim: int, learning_rate: float,
        pred_clipping: float, weight_clipping: float,
        classes: int = None, bias: bool = True, context_bias: bool = True
    ):
        super().__init__(learning_rate, weight_clipping)

    def predict(self, preds, input, target=None):
        ...
        if target is not None:
            ... self.learning_rate, self.weight_clipping ...
        return preds


class Neuron(OnlineUpdateModule):


def GLN(OnlineUpdateModule):

    def __init__(
        layer_sizes: Sequence[int], input_size: int,
        context_dim: int = 4, learning_rate: float = 1e-4, pred_clipping: float = 0.05,
        weight_clipping: float = 5.0, classes: int = None, base_preds: int = None, seed: int = None
    ):
        self.pred_clipping = pred_clipping
        self.classes = classes
        self.base_preds = base_preds
        self.seed = seed

        self.layers = list()
        if self.base_preds is None:
            preds_size = input_size
        else:
            preds_size = base_preds
        for size in layer_sizes:
            self.layers.append(Linear(
                size=size, preds_size=preds_size, input_size=input_size, context_dim=context_dim,
                learning_rate=learning_rate, pred_clipping=pred_clipping,
                weight_clipping=weight_clipping, classes=classes
            ))

    def predict(self, input, target=None):
        if self.base_preds is None:
            preds = jnp.clip(input, a_min=self.pred_clipping, a_max=(1.0 - self.pred_clipping))
        else:
            preds = jnr.uniform(
                key=rng??, shape=base_preds, minval=self.pred_clipping,
                maxval=(1.0 - self.pred_clipping)
            )

        for layer in self.layers:
            preds = layer.predict(preds, input, target=target)

        return preds
