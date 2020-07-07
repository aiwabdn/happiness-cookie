import jax
from jax import lax, nn as jnn, numpy as jnp, random as jnr, scipy as jsp
from numpy import ndarray
from random import randrange
from typing import Callable, Optional, Sequence, Union

from base import GLNBase


jax.config.update("jax_debug_nans", True)
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_numpy_rank_promotion", "raise")


class DynamicParameter(object):

    def initialize(self):
        return 0.0

    def value(self, step):
        return step + 1.0


class ConstantParameter(DynamicParameter):

    def __init__(self, constant_value):
        DynamicParameter.__init__(self)

        assert isinstance(constant_value, float)
        self.constant_value = constant_value

    def value(self, step):
        return super().value(step), self.constant_value


class PaperLearningRate(DynamicParameter):

    def value(self, step):
        step = super().value(step)
        return step, jnp.minimum(100.0 / step, 0.01)


class OnlineUpdateModule(object):

    def __init__(
        self,
        learning_rate: DynamicParameter,
        pred_clipping: float,
        weight_clipping: float
    ):
        assert isinstance(learning_rate, DynamicParameter)
        assert 0.0 < pred_clipping < 1.0
        assert weight_clipping >= 1.0

        self.learning_rate = learning_rate
        self.pred_clipping = pred_clipping
        self.weight_clipping = weight_clipping

    def initialize(self, rng):
        params = dict()
        params['lr_step'] = self.learning_rate.initialize()
        return params

    def predict(self, params, logits, context, target=None):
        raise NotImplementedError()


class Linear(OnlineUpdateModule):

    def __init__(
        self,
        size: int,
        input_size: int,
        context_size: int,
        context_map_size: int,
        classes: int,
        learning_rate: DynamicParameter,
        pred_clipping: float,
        weight_clipping: float,
        bias: bool,
        context_bias: bool
    ):
        super().__init__(learning_rate, pred_clipping, weight_clipping)

        assert size > 0 and input_size > 0 and context_size > 0
        assert context_map_size >= 2
        assert classes >= 1

        self.size = size
        self.input_size = input_size
        self.context_size = context_size
        self.context_map_size = context_map_size
        self.classes = classes
        self.bias = bias
        self.context_bias = context_bias

    def initialize(self, rng):
        rng, rng1 = jnr.split(key=rng, num=2)
        params = super().initialize(rng=rng1)

        logits_size = self.input_size + int(self.bias)
        num_context_indices = 1 << self.context_map_size
        weights_shape = (self.classes, self.size, num_context_indices, logits_size)
        params['weights'] = jnp.full(shape=weights_shape, fill_value=(1.0 / logits_size))

        if self.bias:
            rng, rng1 = jnr.split(key=rng, num=2)
            bias_shape = (1, self.classes, 1)
            params['bias'] = jnr.uniform(
                key=rng1, shape=bias_shape, minval=jsp.special.logit(self.pred_clipping),
                maxval=jsp.special.logit(1.0 - self.pred_clipping)
            )

        context_maps_shape = (1, self.classes, self.size, self.context_map_size, self.context_size)
        if self.context_bias:
            rng1, rng2 = jnr.split(key=rng, num=2)
            context_maps = jnr.normal(key=rng1, shape=context_maps_shape)
            norm = jnp.linalg.norm(context_maps, axis=-1, keepdims=True)
            params['context_maps'] = context_maps / norm

            context_bias_shape = (1, self.classes, self.size, self.context_map_size)
            params['context_bias'] = jnr.normal(key=rng2, shape=context_bias_shape)

        else:
            params['context_maps'] = jnr.normal(key=rng, shape=context_maps_shape)

        return params

    def predict(self, params, logits, context, target=None):
        context = jnp.expand_dims(jnp.expand_dims(jnp.expand_dims(context, axis=1), axis=1), axis=1)
        context_bias = params.get('context_bias', 0.0)
        context_index = (params['context_maps'] * context).sum(axis=-1) > context_bias

        context_map_values = jnp.asarray([[[[1 << n for n in range(self.context_map_size)]]]])
        context_index = jnp.where(context_index, context_map_values, 0)
        context_index = context_index.sum(axis=-1, keepdims=True)

        batch_size = logits.shape[0]
        class_neuron_index = jnp.asarray(
            [[[[c, n] for n in range(self.size)] for c in range(self.classes)]]
        )
        class_neuron_index = jnp.tile(class_neuron_index, reps=(batch_size, 1, 1, 1))
        context_index = jnp.concatenate([class_neuron_index, context_index], axis=-1)

        dims = lax.GatherDimensionNumbers(
            offset_dims=(3,), collapsed_slice_dims=(0, 1, 2), start_index_map=(0, 1, 2)
        )
        weights = lax.gather(
            operand=params['weights'], start_indices=context_index, dimension_numbers=dims,
            slice_sizes=(1, 1, 1, self.input_size + int(self.bias))
        )

        bias = jnp.tile(params['bias'], reps=(batch_size, 1, 1))
        logits = jnp.concatenate([logits, bias], axis=-1)
        logits = jnp.expand_dims(logits, axis=-1)

        output_logits = jnp.matmul(weights, logits)
        output_logits = jnp.clip(
            output_logits, a_min=jsp.special.logit(self.pred_clipping),
            a_max=jsp.special.logit(1.0 - self.pred_clipping)
        )

        if target is None:
            return jnp.squeeze(output_logits, axis=-1)

        else:
            logits = jnp.expand_dims(jnp.squeeze(logits, axis=-1), axis=-2)
            output_preds = jnn.sigmoid(output_logits)
            target = jnp.expand_dims(jnp.expand_dims(target, axis=-1), axis=-1)
            params['lr_step'], learning_rate = self.learning_rate.value(params['lr_step'])
            delta = learning_rate * (target - output_preds) * logits

            dims = lax.ScatterDimensionNumbers(
                update_window_dims=(3,), inserted_window_dims=(0, 1, 2),
                scatter_dims_to_operand_dims=(0, 1, 2)
            )

            if self.weight_clipping is None:
                params['weights'] = lax.scatter_add(
                    operand=params['weights'], scatter_indices=context_index, updates=delta,
                    dimension_numbers=dims
                )
            else:
                weights = jnp.clip(
                    weights + delta, a_min=-self.weight_clipping, a_max=self.weight_clipping
                )
                params['weights'] = lax.scatter(
                    operand=params['weights'], scatter_indices=context_index, updates=weights,
                    dimension_numbers=dims
                )

            return params, jnp.squeeze(output_logits, axis=-1)


class GLN(GLNBase):

    def __init__(
        self,
        layer_sizes: Sequence[int],
        input_size: int,
        context_map_size: int = 4,
        classes: Optional[Union[int, Sequence[object]]] = None,
        base_predictor: Optional[Callable[[ndarray], ndarray]] = None,
        learning_rate: float = 1e-4,
        pred_clipping: float = 1e-3,
        weight_clipping: float = 5.0,
        bias: bool = True,
        context_bias: bool = True,
        seed: Optional[int] = None
    ):
        super().__init__(
            layer_sizes, input_size, context_map_size, classes, base_predictor,
            learning_rate, pred_clipping, weight_clipping, bias, context_bias
        )

        # Learning rate as dynamic parameter
        if self.learning_rate == 'paper':
            self.learning_rate = PaperLearningRate()
        else:
            self.learning_rate = ConstantParameter(self.learning_rate)

        # Random seed
        if seed is None:
            self.seed = randrange(1000000)
        else:
            self.seed = seed
        self.params = dict()
        self.params['rng'] = jnr.PRNGKey(seed=self.seed)

        # Initialize layers
        self.layers = list()
        self.params['rng'], *rngs = jnr.split(
            key=self.params['rng'], num=(len(self.layer_sizes) + 1)
        )
        previous_size = self.base_pred_size
        for n, (size, rng) in enumerate(zip(self.layer_sizes, rngs)):
            layer = Linear(
                size=size, input_size=previous_size, context_size=self.input_size,
                context_map_size=self.context_map_size, classes=self.num_classes,
                learning_rate=self.learning_rate, pred_clipping=self.pred_clipping,
                weight_clipping=self.weight_clipping, bias=self.bias, context_bias=self.context_bias
            )
            self.layers.append(layer)
            self.params[f'layer{n}'] = layer.initialize(rng=rng)
            previous_size = size

        # JAX-compiled predict function
        self._jax_predict = jax.jit(fun=self._predict)

        # JAX-compiled update function
        self._jax_update = jax.jit(fun=self._predict)

    def predict(self, input, target=None):
        # Base predictions
        base_preds = self.base_predictor(input)
        base_preds = jnp.asarray(base_preds, dtype=float)

        # Context
        context = jnp.asarray(input, dtype=float)

        if target is None:
            # Predict without update
            prediction = self._jax_predict(
                params=self.params, base_preds=base_preds, context=context
            )

        else:
            # Target
            if self.num_classes == 1:
                target = jnp.asarray(target, dtype=bool)
            elif self.classes is None:
                target = jnp.asarray(target, dtype=int)
            else:
                target = jnp.asarray([self.classes.index(x) for x in target], dtype=int)

            # Predict with update
            self.params, prediction = self._jax_update(
                params=self.params, base_preds=base_preds, context=input, target=target
            )

        # Predicted class
        if self.classes is None:
            return prediction
        else:
            return [self.classes[x] for x in prediction]

    def _predict(self, params, base_preds, context, target=None):
        # Base logits
        base_preds = jnp.clip(
            base_preds, a_min=self.pred_clipping, a_max=(1.0 - self.pred_clipping)
        )
        logits = jsp.special.logit(base_preds)
        logits = jnp.expand_dims(logits, axis=1)
        logits = jnp.tile(logits, reps=(1, self.num_classes, 1))

        # Turn target class into one-hot
        if target is not None:
            if self.num_classes == 1:
                target = jnp.expand_dims(jnp.where(target, 1.0, 0.0), axis=1)
            else:
                target = jnn.one_hot(target, num_classes=self.num_classes)

        # Layers
        if target is None:
            for n, layer in enumerate(self.layers):
                logits = layer.predict(params=params[f'layer{n}'], logits=logits, context=context)
        else:
            for n, layer in enumerate(self.layers):
                params[f'layer{n}'], logits = layer.predict(
                    params=params[f'layer{n}'], logits=logits, context=context, target=target
                )
        logits = jnp.squeeze(logits, axis=-1)

        # Output prediction
        if self.num_classes == 1:
            prediction = jnp.squeeze(logits, axis=-1) > 0.0
        else:
            prediction = jnp.argmax(logits, axis=1)

        if target is None:
            return prediction
        else:
            return params, prediction

    def evaluate(self, inputs, targets, batch_size):
        assert inputs.shape[0] % batch_size == 0

        inputs = jnp.asarray(inputs)
        targets = jnp.asarray(targets)
        num_instances = inputs.shape[0]

        params = self.params
        self.params = None

        @jax.jit
        def body(n, num_correct):
            # jnp.arange not working here
            # batch = jnp.arange(n * batch_size, (n + 1) * batch_size)
            batch = jnp.linspace(n * batch_size, (n + 1) * batch_size - 1, batch_size, dtype=int)
            batch = batch % num_instances
            prediction = self._jax_predict(params=params, input=inputs[batch])
            num_correct += jnp.count_nonzero(prediction == targets[batch])
            return num_correct

        num_iterations = num_instances // batch_size
        num_correct = lax.fori_loop(lower=0, upper=num_iterations, body_fun=body, init_val=0)

        assert self.params is None
        self.params = params

        return num_correct / num_instances

    def train(self, inputs, targets, batch_size, num_iterations=None, num_epochs=None):
        assert (num_iterations is None) is not (num_epochs is None)
        assert inputs.shape[0] % batch_size == 0

        inputs = jnp.asarray(inputs)
        targets = jnp.asarray(targets)
        num_instances = inputs.shape[0]

        @jax.jit
        def body(n, params):
            if num_epochs is None:
                params['rng'], rng = jnr.split(key=params.pop('rng'), num=2)
                batch = jnr.randint(key=rng, shape=(batch_size,), minval=0, maxval=num_instances)
            else:
                # jnp.arange not working here
                # batch = jnp.arange(n * batch_size, (n + 1) * batch_size)
                batch = jnp.linspace(
                    n * batch_size, (n + 1) * batch_size - 1, num=batch_size, dtype=int
                )
                batch = batch % num_instances
            params, _ = self._jax_update(params=params, base_preds=inputs[batch], target=targets[batch])
            return params

        params = self.params
        self.params = None

        if num_epochs is not None:
            num_iterations = (num_epochs * num_instances) // batch_size
        params = lax.fori_loop(lower=0, upper=num_iterations, body_fun=body, init_val=params)

        assert self.params is None
        self.params = params


def main():
    import numpy as np
    import time
    import utils

    train_images, train_labels, test_images, test_labels = utils.get_mnist()

    model = GLN(
        layer_sizes=[16, 16, 16, 1], input_size=train_images.shape[1], context_map_size=4,
        classes=10, base_predictor=None, learning_rate=1e-4, pred_clipping=1e-3,
        weight_clipping=5.0, bias=True, context_bias=True
    )

    count_weights = (lambda count, x: count + x.size if isinstance(x, jnp.ndarray) else count)
    print('Weights:', jax.tree_util.tree_reduce(count_weights, model.params, initializer=0))

    num_correct = 0
    for n in range(test_images.shape[0] // 100):
        prediction = model.predict(test_images[n * 100: (n + 1) * 100])
        num_correct += np.count_nonzero(prediction == test_labels[n * 100: (n + 1) * 100])
    print('Accuracy:', num_correct / test_images.shape[0])

    start = time.time()
    num_epochs = 1
    batch_size = 10
    for n in range((num_epochs * train_images.shape[0]) // batch_size):
        indices = np.arange(n * batch_size, (n + 1) * batch_size) % train_images.shape[0]
        model.predict(train_images[indices], train_labels[indices])
    print('Time:', time.time() - start)

    num_correct = 0
    for n in range(test_images.shape[0] // 100):
        prediction = model.predict(test_images[n * 100: (n + 1) * 100])
        num_correct += np.count_nonzero(prediction == test_labels[n * 100: (n + 1) * 100])
    print('Accuracy:', num_correct / test_images.shape[0])


if __name__ == '__main__':
    main()
