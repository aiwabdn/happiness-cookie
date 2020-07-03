import jax
from jax import lax, nn as jnn, numpy as jnp, random as jnr, scipy as jsp
from typing import Sequence


jax.config.update("jax_enable_x64", True)
jax.config.update("jax_numpy_rank_promotion", "raise")
# jax.config.update("jax_debug_nans", True)


class OnlineUpdateModule(object):

    def __init__(self, learning_rate: float, pred_clipping: float, weight_clipping: float):
        assert learning_rate > 0.0
        assert 0.0 < pred_clipping < 1.0
        assert weight_clipping is None or weight_clipping > 1.0

        self.learning_rate = learning_rate
        self.pred_clipping = pred_clipping
        self.weight_clipping = weight_clipping

    def initialize(self, rng):
        # return params
        raise NotImplementedError()

    def predict(self, logits, context, target=None):
        # return logits
        raise NotImplementedError()


class Linear(OnlineUpdateModule):

    def __init__(
        self, size: int, input_size: int, context_size: int, context_map_size: int,
        learning_rate: float, pred_clipping: float, weight_clipping: float,
        classes: int = None, bias: bool = True, context_bias: bool = True
    ):
        super().__init__(learning_rate, pred_clipping, weight_clipping)

        assert size > 0 and input_size > 0 and context_size > 0
        assert context_map_size >= 2
        assert classes is None or classes >= 2

        self.size = size
        self.input_size = input_size
        self.context_size = context_size
        self.context_map_size = context_map_size
        self.classes = classes
        self.bias = bias
        self.context_bias = context_bias

    def initialize(self, rng):
        logits_size = self.input_size + int(self.bias)
        num_context_indices = 1 << self.context_map_size
        if self.classes is None:
            context_maps_shape = (1, self.size, self.context_map_size, self.context_size)
            context_bias_shape = (1, self.size, self.context_map_size)
            weights_shape = (self.size, num_context_indices, logits_size)
            bias_shape = (1, 1)
        else:
            context_maps_shape = (1, self.classes, self.size, self.context_map_size, self.context_size)
            context_bias_shape = (1, self.classes, self.size, self.context_map_size)
            weights_shape = (self.classes, self.size, num_context_indices, logits_size)
            bias_shape = (1, self.classes, 1)

        params = dict()
        params['weights'] = jnp.full(shape=weights_shape, fill_value=(1.0 / logits_size))
        if self.bias:
            rng, rng1 = jnr.split(key=rng, num=2)
            params['bias'] = jnr.uniform(
                key=rng1, shape=bias_shape, minval=jsp.special.logit(self.pred_clipping),
                maxval=jsp.special.logit(1.0 - self.pred_clipping)
            )

        if self.context_bias:
            rng1, rng2 = jnr.split(key=rng, num=2)
            context_maps = jnr.normal(key=rng1, shape=context_maps_shape)
            norm = jnp.linalg.norm(context_maps, axis=-1, keepdims=True)
            params['context_maps'] = context_maps / norm
            params['context_bias'] = jnr.normal(key=rng2, shape=context_bias_shape)
        else:
            params['context_maps'] = jnr.normal(key=rng, shape=context_maps_shape)

        return params

    def predict(self, logits, context, target=None):
        context = jnp.expand_dims(jnp.expand_dims(context, axis=1), axis=1)
        if self.classes is not None:
            context = jnp.expand_dims(context, axis=1)

        if 'context_bias' in self.params:
            context_index = (context_maps * context).sum(axis=-1) > context_bias
        else:
            context_index = (context_maps * context).sum(axis=-1) > 0.0

        if self.classes is None:
            context_map_values = jnp.asarray([[[1 << n for n in range(self.context_map_size)]]])
        else:
            context_map_values = jnp.asarray([[[[1 << n for n in range(self.context_map_size)]]]])
        context_index = jnp.where(context_index, context_map_values, 0)
        context_index = context_index.sum(axis=-1, keepdims=True)

        weights = jnp.take_along_axis(self.params['weights'], indices=context_index, axis=0)???

        batch_size = logits.shape[0]
        if self.classes is None:
            bias = jnp.tile(self.params['bias'], reps=(batch_size, 1))
        else:
            bias = jnp.tile(self.params['bias'], reps=(batch_size, 1, 1))
        logits = jnp.concatenate([logits, bias], axis=-1)
        logits = jnp.expand_dims(logits, axis=-1)

        output_logits = jnp.matmul(weights, logits)
        output_logits = jnp.clip(
            output_logits, a_min=jsp.special.logit(self.pred_clipping),
            a_max=jsp.special.logit(1.0 - self.pred_clipping)
        )

        if target is not None:
            logits = jnp.expand_dims(jnp.squeeze(logits, axis=-1), axis=-2)
            output_preds = jnn.sigmoid(output_logits)
            target = jnp.expand_dims(jnp.expand_dims(target, axis=-1), axis=-1)
            delta = self.learning_rate * (target - output_preds) * logits
            # context_index = jnp.squeeze(context_index, axis=-1)
            # delta = jax.ops.segment_sum(data=delta, segment_ids=context_index, num_segments=(1 << self.context_map_size))
            # self.params['weights'] += delta
            # params['weights'] = jax.ops.index_add(params['weights'], contexts, delta)

            # context_index = jnp.expand_dims(context_index, axis=3)
            # delta = jnp.expand_dims(delta, axis=3)
            context_index = jnp.concatenate([context_index, jnp.tile(jnp.expand_dims(jnp.expand_dims(jnp.arange(self.size), axis=1), axis=0), (logits.shape[0], 1, 1))], axis=2)

            # delta = jnp.expand_dims(delta, axis=1)
            if self.classes is None:
                dims = lax.ScatterDimensionNumbers(
                    # (2,)
                    update_window_dims=(2,), inserted_window_dims=(0, 1),
                    scatter_dims_to_operand_dims=(0, 1)
                )
            else:
                dims = lax.ScatterDimensionNumbers(
                    update_window_dims=(1, 2), inserted_window_dims=(),
                    scatter_dims_to_operand_dims=(0, 3)
                )
            self.params['weights'] = lax.scatter_add(
                operand=self.params['weights'], scatter_indices=context_index, updates=delta,
                dimension_numbers=dims
            )
            # if self.weight_clipping is None:
            #     self.weights.scatter_nd_add(indices=neuron_context_index, updates=delta)
            # else:
            #     weights = tf.clip_by_value(
            #         weights + delta, clip_value_min=-self.weight_clipping,
            #         clip_value_max=self.weight_clipping
            #     )
            #     self.weights.scatter_nd_update(indices=neuron_context_index, updates=weights)

            if self.weight_clipping is not None:
                self.params['weights'] = jnp.clip(
                    self.params['weights'], a_min=-self.weight_clipping, a_max=self.weight_clipping
                )

        return jnp.squeeze(output_logits, axis=-1)


class GLN(OnlineUpdateModule):

    def __init__(
        self, layer_sizes: Sequence[int], input_size: int,
        context_map_size: int = 4, learning_rate: float = 1e-4, pred_clipping: float = 0.05,
        weight_clipping: float = None, classes: int = None, base_preds: int = None, seed: int = 0
    ):
        super().__init__(learning_rate, pred_clipping, weight_clipping)

        assert len(layer_sizes) > 0 and layer_sizes[-1] == 1
        assert input_size > 0
        assert context_map_size >= 2
        assert classes is None or classes >= 2
        assert base_preds is None or base_preds > 0

        self.pred_clipping = pred_clipping
        self.classes = classes
        self.base_preds = base_preds
        self.seed = seed

        if self.base_preds is None:
            logits_size = input_size
        else:
            logits_size = base_preds

        self.layers = list()
        for size in layer_sizes:
            self.layers.append(Linear(
                size=size, input_size=logits_size, context_size=input_size,
                context_map_size=context_map_size, learning_rate=learning_rate,
                pred_clipping=pred_clipping, weight_clipping=weight_clipping, classes=classes
            ))
            logits_size = size

    def initialize(self):
        self.params = dict()
        self.params['rng'] = jnr.PRNGKey(seed=self.seed)

        # Base predictions
        if self.base_preds is not None:
            self.params['rng'], rng = jnr.split(key=self.params['rng'], num=2)
            if self.classes is None:
                base_logits_shape = (1, self.base_preds)
            else:
                base_logits_shape = (1, self.classes, self.base_preds)
            self.params['base_logits'] = jnr.uniform(
                key=rng, shape=base_logits_shape,
                minval=jsp.special.logit(self.pred_clipping),
                maxval=jsp.special.logit(1.0 - self.pred_clipping)
            )

        # Initialize layers
        self.params['rng'], *rngs = jnr.split(key=self.params['rng'], num=(len(self.layers) + 1))
        for n, (layer, rng) in enumerate(zip(self.layers, rngs)):
            self.params[f'layer{n}'] = layer.initialize(rng=rng)

        # JAX-compiled predict function
        # @jax.jit
        def _jax_predict(params, input):
            self.params = params
            for n, layer in enumerate(self.layers):
                layer.params = self.params[f'layer{n}']
            return self._predict(input)

        # JAX-compiled update function
        # @jax.jit
        def _jax_update(params, input, target):
            self.params = params
            for n, layer in enumerate(self.layers):
                layer.params = self.params[f'layer{n}']
            prediction = self._predict(input, target)
            for n, layer in enumerate(self.layers):
                self.params[f'layer{n}'] = layer.params
            return self.params, prediction

        self._jax_predict = _jax_predict
        self._jax_update = _jax_update

    def predict(self, input, target=None):
        params = dict(self.params)
        if target is None:  # predict
            pred = self._jax_predict(params=params, input=input)
        else:  # predict with online update
            params, pred = self._jax_update(params=params, input=input, target=target)
        self.params = dict(params)
        return pred

    def _predict(self, input, target=None):
        # Base predictions
        if 'base_logits' in self.params:
            batch_size = input.shape[0]
            if self.classes is None:
                logits = jnp.tile(self.params['base_logits'], reps=(batch_size, 1))
            else:
                logits = jnp.tile(self.params['base_logits'], reps=(batch_size, 1, 1))
        else:
            logits = jnp.clip(input, a_min=self.pred_clipping, a_max=(1.0 - self.pred_clipping))
            logits = jsp.special.logit(logits)
            if self.classes is not None:
                logits = jnp.expand_dims(logits, axis=1)
                logits = jnp.tile(logits, reps=(1, self.classes, 1))

        # Turn class integer into one-hot
        if target is not None:
            if self.classes is not None and jax.dtypes.issubdtype(target.dtype, jnp.integer):
                target = jnn.one_hot(target, num_classes=self.classes, dtype=bool)
            else:
                target = jnp.where(target, 1.0, 0.0)

        # Layers
        for n, layer in enumerate(self.layers):
            logits = layer.predict(logits=logits, context=input, target=target)

        # Output prediction
        logits = jnp.squeeze(logits, axis=-1)
        if self.classes is None:
            return logits > 0.0
        else:
            return jnp.argmax(logits, axis=1)


# # @functools.partial(jax.jit, static_argnums=(4,))
# def evaluate(model, params, inputs, targets, batch_size):
#     assert inputs.shape[0] % batch_size == 0

#     inputs = jnp.asarray(inputs)
#     targets = jnp.asarray(targets)
#     num_instances = inputs.shape[0]

#     output = model.predict(params=params, input=inputs[:batch_size])
#     for n in range(1, num_instances // batch_size):
#         x = model.predict(params=params, input=inputs[n * batch_size: (n + 1) * batch_size])
#         output = jnp.concatenate((output, x), axis=0)

#     @jax.jit
#     def body_fn(n, accuracy):
#         start = n * batch_size
#         stop = (n + 1) * batch_size
#         try:
#             batch = jnp.arange(start, stop) % num_instances
#         except BaseException:
#             print('jnp.arange still not working!')
#             batch = jnp.linspace(start, stop, num=batch_size, dtype=int)
#         prediction = model.predict(params=params, input=inputs[batch])
#         accuracy += jnp.count_nonzero(prediction == targets[batch])
#         return accuracy

#     num_iterations = num_instances // batch_size
#     accuracy = lax.fori_loop(lower=0, upper=num_iterations, body_fun=body_fn, init_val=0.0)
#     return accuracy / num_iterations


# # @functools.partial(jax.jit, static_argnums=(4,))
# def train(model, params, inputs, targets, batch_size, num_iterations=None, num_epochs=None):
#     assert (num_iterations is None) is not (num_epochs is None)
#     assert inputs.shape[0] % batch_size == 0

#     inputs = jnp.asarray(inputs)
#     targets = jnp.asarray(targets)
#     num_instances = inputs.shape[0]

#     @jax.jit
#     def body_fn(n, args):
#         params, loss = args
#         if num_epochs is None:
#             params['rng'], rng = jnr.split(key=params.pop('rng'), num=2)
#             batch = jnr.randint(key=rng, shape=(batch_size,), minval=0, maxval=num_instances)
#         else:
#             start = n * batch_size
#             stop = (n + 1) * batch_size
#             try:
#                 batch = jnp.arange(start, stop) % num_instances
#             except BaseException:
#                 print('jnp.arange still not working!')
#                 batch = jnp.linspace(start, stop, num=batch_size, dtype=int)
#         params, _loss = model.update(params=params, input=inputs[batch], target=targets[batch])
#         loss = 0.9 * loss + 0.1 * _loss
#         return params, loss

#     if num_epochs is not None:
#         num_iterations = num_instances // batch_size
#     params, loss = lax.fori_loop(
#         lower=0, upper=num_iterations, body_fun=body_fn, init_val=(params, 0.0)
#     )
#     return params, loss


def main():
    import numpy as np
    import time
    from test_mnist import get_mnist_numpy

    train_images, train_labels, test_images, test_labels = get_mnist_numpy()

    train_labels = train_labels == 1
    test_labels = test_labels == 1

    model = GLN(
        layer_sizes=[4, 4, 1], input_size=train_images.shape[1], context_map_size=4,
        learning_rate=1e-3, pred_clipping=0.05, weight_clipping=5.0, classes=None, base_preds=4
    )

    model.initialize()

    count_weights = (lambda count, x: count + x.size if isinstance(x, jnp.ndarray) else count)
    print('Weights:', jax.tree_util.tree_reduce(count_weights, model.params, initializer=0))

    print('Accuracy:', np.count_nonzero(model.predict(test_images) == test_labels) / test_labels.shape[0])
    start = time.time()
    batch_size = 10
    for n in range(100):  # train_labels.shape[0] // batch_size
        input = train_images[n * batch_size: (n + 1) * batch_size]
        target = train_labels[n * batch_size: (n + 1) * batch_size]
        model.predict(input, target)
    print('Time:', time.time() - start)
    print('Accuracy:', np.count_nonzero(model.predict(test_images) == test_labels) / test_labels.shape[0])


if __name__ == '__main__':
    main()
