from collections import namedtuple
import jax
from jax import lax, nn as jnn, numpy as jnp, random as jnr, scipy as jsp
import time
from typing import Sequence


jax.config.update("jax_enable_x64", True)
jax.config.update("jax_numpy_rank_promotion", "raise")
# jax.config.update("jax_debug_nans", True)


UpdateTransformation = namedtuple(
    typename='UpdateTransformation', field_names=('init', 'apply', 'update', 'spec')
)
Loss = namedtuple(typename='Loss', field_names=('init', 'predict', 'loss', 'spec'))
Model = namedtuple(typename='Model', field_names=('init', 'predict', 'update', 'spec'))


def Linear(
    size: int, context: int, classes: int = None, bias: bool = True, context_bias: bool = True,
    epsilon: float = 0.05
):
    assert size >= 0
    assert classes is None or classes >= 2
    assert context >= 1
    assert 0.0 < epsilon < 1.0

    def init_fn(rng, input_spec):
        assert input_spec['info']['type'] == 'float' and len(input_spec['info']['shape']) == 1
        if classes is None:
            assert input_spec['logits']['type'] == 'float' and len(input_spec['logits']['shape']) == 1
        else:
            assert input_spec['logits']['type'] == 'float' and len(input_spec['logits']['shape']) == 2

        params = dict()
        if bias and context_bias:
            rng1, rng2, rng3 = jnr.split(key=rng, num=3)
        elif bias:
            rng1, rng3 = jnr.split(key=rng, num=2)
        elif context_bias:
            rng1, rng2 = jnr.split(key=rng, num=2)
        else:
            rng1 = rng

        if classes is None:
            if size == 0:
                logits_spec = dict(type='float', shape=())
            else:
                logits_spec = dict(type='float', shape=(size,))
            output_spec = dict(logits=logits_spec, info=dict(input_spec['info']))
        else:
            if size == 0:
                logits_spec = dict(type='float', shape=(classes,))
            else:
                logits_spec = dict(type='float', shape=(classes, size))
            output_spec = dict(logits=logits_spec, info=dict(input_spec['info']))

        logits_size = input_spec['logits']['shape'][-1] + int(bias)
        info_size = input_spec['info']['shape'][-1]
        context_size = 1 << context

        if classes is None:
            contexts_shape = (max(1, size), context, info_size)
            contexts_bias_shape = (max(1, size), context)
            weights_shape = (context_size, max(1, size), logits_size)
            bias_shape = (1, 1)
        else:
            contexts_shape = (classes, max(1, size), context, info_size)
            contexts_bias_shape = (classes, max(1, size), context)
            weights_shape = (context_size, classes, max(1, size), logits_size)
            bias_shape = (1, classes, 1)

        if context_bias:
            contexts = jnr.normal(key=rng1, shape=contexts_shape)
            params['contexts'] = contexts / jnp.linalg.norm(contexts, axis=-1, keepdims=True)
            params['contexts_bias'] = jnr.normal(key=rng2, shape=contexts_bias_shape)
        else:
            params['contexts'] = jnr.normal(key=rng1, shape=contexts_shape)

        params['weights'] = jnp.full(shape=weights_shape, fill_value=(1.0 / logits_size))

        if bias:
            params['bias'] = jnr.uniform(
                key=rng3, shape=bias_shape, minval=epsilon, maxval=(1.0 - epsilon)
            )

        return output_spec, params

    def apply_fn(rng, params, input):
        contexts = jnp.expand_dims(params['contexts'], axis=0)
        info = jnp.expand_dims(jnp.expand_dims(input['info'], axis=1), axis=1)
        if classes is not None:
            info = jnp.expand_dims(info, axis=1)
        if context_bias:
            contexts_bias = jnp.expand_dims(params['contexts_bias'], axis=0)
            contexts = (contexts * info).sum(axis=-1) > contexts_bias
        else:
            contexts = (contexts * info).sum(axis=-1) > 0.0
        context_values = jnp.asarray([[[1 << n for n in range(context)]]], dtype=int)
        if classes is not None:
            context_values = jnp.expand_dims(context_values, axis=1)
        contexts = jnp.where(contexts, context_values, 0).sum(axis=-1, keepdims=True)

        weights = jnp.take_along_axis(params['weights'], indices=contexts, axis=0)

        input_logits = input['logits']
        if classes is None:
            bias_pred = jnp.tile(params['bias'], reps=(input_logits.shape[0], 1))
        else:
            bias_pred = jnp.tile(params['bias'], reps=(input_logits.shape[0], 1, 1))
        input_logits = jnp.concatenate([input_logits, bias_pred], axis=-1)
        input_logits = jnp.expand_dims(input_logits, axis=-1)

        output_logits = jnp.matmul(weights, input_logits)
        output_logits = jnp.squeeze(output_logits, axis=-1)
        output_logits = jnp.clip(
            output_logits, a_min=jsp.special.logit(epsilon), a_max=jsp.special.logit(1.0 - epsilon)
        )

        if size == 0:
            output_logits = output_logits.squeeze(axis=-1)

        return dict(logits=output_logits, info=input['info'])

    def update_fn(rng, params, input, update):
        contexts = jnp.expand_dims(params['contexts'], axis=0)
        info = jnp.expand_dims(jnp.expand_dims(input['info'], axis=1), axis=1)
        if classes is not None:
            info = jnp.expand_dims(info, axis=1)
        if context_bias:
            contexts_bias = jnp.expand_dims(params['contexts_bias'], axis=0)
            contexts = (contexts * info).sum(axis=-1) > contexts_bias
        else:
            contexts = (contexts * info).sum(axis=-1) > 0.0
        context_values = jnp.asarray([[[1 << n for n in range(context)]]], dtype=int)
        if classes is not None:
            context_values = jnp.expand_dims(context_values, axis=1)
        contexts = jnp.where(contexts, context_values, 0).sum(axis=-1, keepdims=True)

        weights = jnp.take_along_axis(params['weights'], indices=contexts, axis=0)

        input_logits = input['logits']
        if classes is None:
            bias_pred = jnp.tile(params['bias'], reps=(input_logits.shape[0], 1))
        else:
            bias_pred = jnp.tile(params['bias'], reps=(input_logits.shape[0], 1, 1))
        input_logits = jnp.concatenate([input_logits, bias_pred], axis=-1)
        input_logits = jnp.expand_dims(input_logits, axis=-1)

        output_logits = jnp.matmul(weights, input_logits)
        output_logits = jnp.clip(
            output_logits, a_min=jsp.special.logit(epsilon), a_max=jsp.special.logit(1.0 - epsilon)
        )

        target, learning_rate, clip_weights = update
        input_logits = jnp.expand_dims(jnp.squeeze(input_logits, axis=-1), axis=-2)
        output_preds = jnn.sigmoid(output_logits)
        target = jnp.expand_dims(jnp.expand_dims(target, axis=-1), axis=-1)
        delta = learning_rate * (target - output_preds) * input_logits
        contexts = jnp.squeeze(contexts, axis=-1)
        # delta = jax.ops.segment_sum(data=delta, segment_ids=contexts, num_segments=(1 << context))
        # params['weights'] += delta
        # params['weights'] = jax.ops.index_add(params['weights'], contexts, delta)

        if classes is None:
            dims = lax.ScatterDimensionNumbers(update_window_dims=(0, 1), inserted_window_dims=(), scatter_dims_to_operand_dims=(0, 1))
        else:
            dims = lax.ScatterDimensionNumbers(update_window_dims=(0,), inserted_window_dims=(), scatter_dims_to_operand_dims=(0, 1, 2))
        params['weights'] = lax.scatter_add(operand=params['weights'], scatter_indices=contexts, updates=delta, dimension_numbers=dims)

        if clip_weights is not None:
            params['weights'] = jnp.clip(params['weights'], a_min=-clip_weights, a_max=clip_weights)
        # params['weights'] += jnp.asarray(full_delta)
        # params['weights'] = jax.lax.scatter_add(
        #     operand=params['weights'], scatter_indices=contexts, updates=delta,
        #     dimension_numbers=1
        # )

        output_logits = jnp.squeeze(output_logits, axis=-1)
        if size == 0:
            output_logits = jnp.squeeze(output_logits, axis=-1)

        return params, dict(logits=output_logits, info=input['info'])

    def spec_fn():
        return dict(
            type='Linear', size=size, context=context, bias=bias, context_bias=context_bias,
            epsilon=epsilon
        )

    return UpdateTransformation(init=init_fn, apply=apply_fn, update=update_fn, spec=spec_fn)


def Sequential(layers: Sequence[UpdateTransformation]):

    def init_fn(rng, input_spec):
        params = dict()
        rngs = jnr.split(key=rng, num=len(layers))
        for n, (layer, rng) in enumerate(zip(layers, rngs)):
            input_spec, params[f'sequential{n}'] = layer.init(rng=rng, input_spec=input_spec)
        return input_spec, params

    def apply_fn(rng, params, input):
        rngs = jnr.split(key=rng, num=len(layers))
        for n, (layer, rng) in enumerate(zip(layers, rngs)):
            input = layer.apply(rng=rng, params=params[f'sequential{n}'], input=input)
        return input

    def update_fn(rng, params, input, update):
        params = dict(params)
        rngs = jnr.split(key=rng, num=len(layers))
        for n, (layer, rng) in enumerate(zip(layers, rngs)):
            params[f'sequential{n}'], input = layer.update(
                rng=rng, params=params[f'sequential{n}'], input=input, update=update
            )
        return params, input

    def spec_fn():
        return dict(type='Sequential', layers=[layer.spec() for layer in layers])

    return UpdateTransformation(init=init_fn, apply=apply_fn, update=update_fn, spec=spec_fn)


def CrossEntropy(binary: bool = False):

    def init_fn(output_spec, target_spec):
        assert output_spec['type'] == 'float'
        assert output_spec['shape'][1:] == target_spec['shape']
        if binary:
            assert target_spec['type'] == 'bool'
            assert output_spec['shape'][0] == 1 or output_spec['shape'][0] == 2
        else:
            assert target_spec['type'] == 'int'
            assert output_spec['shape'][0] >= 2

    def predict_fn(rng, output):
        if output.shape[1] == 1:
            return jnp.squeeze(output, axis=1) > 0.0
        elif binary:
            return output[:, 1] > output[:, 0]
        else:
            return jnp.argmax(output, axis=1)

    def loss_fn(rng, output, target):
        if output.shape[1] == 1:
            true_prob = jnn.sigmoid(jnp.squeeze(output, axis=1))
            target_logit = jnp.where(
                condition=target, x=jnp.log(jnp.maximum(1e-6, true_prob)),
                y=jnp.log(jnp.maximum(1e-6, 1.0 - true_prob))
            )
            num_true = jnp.count_nonzero(target)
            true_weight = (target.size - num_true) / jnp.maximum(1e-6, num_true)
            weight = jnp.where(condition=target, x=true_weight, y=1.0)
        elif binary:
            logits = jnn.log_softmax(output, axis=1)
            target_logit = jnp.where(condition=target, x=logits[:, 1], y=logits[:, 0])
            num_true = jnp.count_nonzero(target)
            true_weight = (target.size - num_true) / jnp.maximum(1e-6, num_true)
            weight = jnp.where(condition=target, x=true_weight, y=1.0)
        else:
            logits = jnn.log_softmax(output, axis=1)
            target_logit = jnp.squeeze(
                jnp.take_along_axis(logits, jnp.expand_dims(target, axis=1), axis=1), axis=1
            )
            values, counts = jnp.unique(target, return_counts=True)
            weights = counts.max() / jnp.maximum(1e-6, counts)
            weight = weights[target]
        loss = -lax.stop_gradient(weight) * target_logit
        return loss.mean(axis=tuple(range(1, loss.ndim))).sum()

    def spec_fn():
        return dict(type='CrossEntropy', binary=binary)

    return Loss(init=init_fn, predict=predict_fn, loss=loss_fn, spec=spec_fn)


def GLN(
    network: UpdateTransformation, loss: Loss, learning_rate: float, classes: int = None,
    clip_weights: float = None, base_preds: int = None, epsilon: float = 0.05, seed: int = 0
):
    assert learning_rate > 0.0
    assert classes is None or classes >= 2
    assert clip_weights is None or clip_weights > 1.0
    assert base_preds is None or base_preds >= 1
    assert 0.0 < epsilon < 1.0

    def init_fn(input_spec, target_spec):
        rng = jnr.PRNGKey(seed=seed)
        if base_preds is None:
            rng1, rng2 = jnr.split(key=rng, num=2)
        else:
            rng1, rng2, rng3 = jnr.split(key=rng, num=3)

        if base_preds is None:
            logits_spec = dict(input_spec)
            if classes is not None:
                logits_spec['shape'] = (classes,) + logits_spec['shape']
        else:
            if classes is None:
                logits_spec = dict(type='float', shape=(base_preds,))
            else:
                logits_spec = dict(type='float', shape=(classes, base_preds))
        input_spec = dict(logits=logits_spec, info=dict(input_spec))

        output_spec, params = network.init(rng=rng1, input_spec=input_spec)
        output_spec = output_spec['logits']

        if classes is None:
            assert target_spec['type'] == 'bool' and target_spec['shape'] == ()
        else:
            assert len(output_spec['shape']) == 2
            output_spec['shape'] = (output_spec['shape'][1], output_spec['shape'][0])
            if target_spec['type'] == 'int':
                assert target_spec['shape'] == ()
                target_spec = dict(type='bool', shape=(classes,))
            else:
                assert target_spec['type'] == 'bool' and target_spec['shape'] == (classes,)
        loss.init(output_spec=output_spec, target_spec=target_spec)

        params = dict(rng=rng2, network=params)
        if base_preds is not None:
            if classes is None:
                params['base_logits'] = jsp.special.logit(jnr.uniform(
                    key=rng3, shape=(1, base_preds), minval=epsilon, maxval=(1.0 - epsilon)
                ))
            else:
                params['base_logits'] = jsp.special.logit(jnr.uniform(
                    key=rng3, shape=(1, classes, base_preds), minval=epsilon, maxval=(1.0 - epsilon)
                ))

        return params

    @jax.jit
    def predict_fn(params, input):
        rng1, rng2 = jnr.split(key=params['rng'], num=2)
        if base_preds is None:
            logits = jsp.special.logit(jnp.clip(input, a_min=epsilon, a_max=(1.0 - epsilon)))
            if classes is not None:
                logits = jnp.expand_dims(logits, axis=1)
                logits = jnp.tile(logits, reps=(1, classes, 1))
        else:
            if classes is None:
                logits = jnp.tile(params['base_logits'], reps=(input.shape[0], 1))
            else:
                logits = jnp.tile(params['base_logits'], reps=(input.shape[0], 1, 1))
        input = dict(logits=logits, info=input)
        output = network.apply(rng=rng1, params=params['network'], input=input)
        if classes is None:
            return loss.predict(rng=rng2, output=output['logits'])
        else:
            return jnp.argmax(jnp.squeeze(output['logits'], axis=2), axis=1)

    @jax.jit
    def update_fn(params, input, target):
        # TODO: deep copy instead of local everywhere
        params = dict(params)
        rng1, rng2 = jnr.split(key=params['rng'], num=2)

        if base_preds is None:
            logits = jsp.special.logit(jnp.clip(input, a_min=epsilon, a_max=(1.0 - epsilon)))
            if classes is not None:
                logits = jnp.expand_dims(logits, axis=1)
                logits = jnp.tile(logits, reps=(1, classes, 1))
        else:
            if classes is None:
                logits = jnp.tile(params['base_logits'], reps=(input.shape[0], 1))
            else:
                logits = jnp.tile(params['base_logits'], reps=(input.shape[0], 1, 1))
        input = dict(logits=logits, info=input)

        if classes is not None and jax.dtypes.issubdtype(target.dtype, jnp.integer):
            target = jnn.one_hot(target, num_classes=classes, dtype=bool)
        update = (target, learning_rate, clip_weights)

        params['network'], output = network.update(rng=rng1, params=params['network'], input=input, update=update)  # TODO: not great since params are updated implicitly
        loss_value = loss.loss(rng=rng2, output=output['logits'], target=target)

        return params, loss_value

    def spec_fn():
        return dict(
            type='GLN', network=network.spec(), loss=loss.spec(), learning_rate=learning_rate,
            clip_weights=clip_weights, base_preds=base_preds, epsilon=epsilon, seed=seed
        )

    return Model(init=init_fn, predict=predict_fn, update=update_fn, spec=spec_fn)


# @functools.partial(jax.jit, static_argnums=(4,))
def evaluate(model, params, inputs, targets, batch_size):
    assert inputs.shape[0] % batch_size == 0

    inputs = jnp.asarray(inputs)
    targets = jnp.asarray(targets)
    num_instances = inputs.shape[0]

    output = model.predict(params=params, input=inputs[:batch_size])
    for n in range(1, num_instances // batch_size):
        x = model.predict(params=params, input=inputs[n * batch_size: (n + 1) * batch_size])
        output = jnp.concatenate((output, x), axis=0)

    @jax.jit
    def body_fn(n, accuracy):
        start = n * batch_size
        stop = (n + 1) * batch_size
        try:
            batch = jnp.arange(start, stop) % num_instances
        except BaseException:
            print('jnp.arange still not working!')
            batch = jnp.linspace(start, stop, num=batch_size, dtype=int)
        prediction = model.predict(params=params, input=inputs[batch])
        accuracy += jnp.count_nonzero(prediction == targets[batch])
        return accuracy

    num_iterations = num_instances // batch_size
    accuracy = lax.fori_loop(lower=0, upper=num_iterations, body_fun=body_fn, init_val=0.0)
    return accuracy / num_iterations


# @functools.partial(jax.jit, static_argnums=(4,))
def train(model, params, inputs, targets, batch_size, num_iterations=None, num_epochs=None):
    assert (num_iterations is None) is not (num_epochs is None)
    assert inputs.shape[0] % batch_size == 0

    inputs = jnp.asarray(inputs)
    targets = jnp.asarray(targets)
    num_instances = inputs.shape[0]

    @jax.jit
    def body_fn(n, args):
        params, loss = args
        if num_epochs is None:
            params['rng'], rng = jnr.split(key=params.pop('rng'), num=2)
            batch = jnr.randint(key=rng, shape=(batch_size,), minval=0, maxval=num_instances)
        else:
            start = n * batch_size
            stop = (n + 1) * batch_size
            try:
                batch = jnp.arange(start, stop) % num_instances
            except BaseException:
                print('jnp.arange still not working!')
                batch = jnp.linspace(start, stop, num=batch_size, dtype=int)
        params, _loss = model.update(params=params, input=inputs[batch], target=targets[batch])
        loss = 0.9 * loss + 0.1 * _loss
        return params, loss

    if num_epochs is not None:
        num_iterations = num_instances // batch_size
    params, loss = lax.fori_loop(
        lower=0, upper=num_iterations, body_fun=body_fn, init_val=(params, 0.0)
    )
    return params, loss


# ----------
# Copied and modified from https://github.com/google/jax/blob/master/examples/datasets.py
# ----------


import array
import gzip
import numpy as np
import os
import struct
import urllib.request


_DATA = "/tmp/mnist_data/"


def get_mnist():
    """Download and parse the raw MNIST dataset."""
    # CVDF mirror of http://yann.lecun.com/exdb/mnist/
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"

    def parse_labels(filename):
        with gzip.open(filename, "rb") as fh:
            _ = struct.unpack(">II", fh.read(8))
            return np.array(array.array("B", fh.read()), dtype=np.uint8)

    def parse_images(filename):
        with gzip.open(filename, "rb") as fh:
            _, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
            return np.array(array.array("B", fh.read()), dtype=np.uint8).reshape(num_data, rows, cols)

    def _download(url, filename):
        """Download a url to a file in the JAX data temp directory."""
        if not os.path.exists(_DATA):
            os.makedirs(_DATA)
        out_file = os.path.join(_DATA, filename)
        if not os.path.isfile(out_file):
            urllib.request.urlretrieve(url, out_file)
            print("downloaded {} to {}".format(url, _DATA))

    for filename in (
        "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz", "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz"
    ):
        _download(base_url + filename, filename)

    train_images = parse_images(os.path.join(_DATA, "train-images-idx3-ubyte.gz")) / 255.0
    train_labels = parse_labels(os.path.join(_DATA, "train-labels-idx1-ubyte.gz"))
    test_images = parse_images(os.path.join(_DATA, "t10k-images-idx3-ubyte.gz")) / 255.0
    test_labels = parse_labels(os.path.join(_DATA, "t10k-labels-idx1-ubyte.gz"))

    return train_images, train_labels, test_images, test_labels


# ----------
#    Main
# ----------


def main():
    network = Sequential(layers=[Linear(size=5, context=4), Linear(size=1, context=3)])
    loss = CrossEntropy(binary=True)
    model = GLN(network=network, loss=loss, learning_rate=1e-4, base_preds=None)
    input_spec = dict(type='float', shape=(8,))
    target_spec = dict(type='bool', shape=())
    params = model.init(input_spec=input_spec, target_spec=target_spec)

    train_images, train_labels, test_images, test_labels = get_mnist()
    train_images = np.reshape(train_images, (-1, 784))
    test_images = np.reshape(test_images, (-1, 784))
    # train_labels = (train_labels == 1)
    # test_labels = (test_labels == 1)
    classes = 10

    network = Sequential(layers=[
        Linear(size=32, context=4, classes=classes, bias=True, context_bias=True, epsilon=0.05),
        Linear(size=32, context=4, classes=classes, bias=True, context_bias=True, epsilon=0.05),
        Linear(size=1, context=4, classes=classes, bias=True, context_bias=True, epsilon=0.05)
    ])
    loss = CrossEntropy(binary=True)
    model = GLN(
        network=network, loss=loss, learning_rate=1e-4, classes=classes, clip_weights=5.0, base_preds=32,
        epsilon=0.05
    )

    input_spec = dict(type='float', shape=train_images.shape[1:])
    target_spec = dict(type='int', shape=train_labels.shape[1:])
    params = model.init(input_spec=input_spec, target_spec=target_spec)

    count_weights = (lambda acc, x: acc + x.size if isinstance(x, jnp.ndarray) else acc)
    print('Weights:', jax.tree_util.tree_reduce(function=count_weights, tree=params, initializer=0))


    print(evaluate(
        model=model, params=params, inputs=test_images, targets=test_labels, batch_size=100
    ))
    start = time.time()
    params, loss = train(
        model=model, params=params, inputs=train_images, targets=train_labels, batch_size=10,
        num_epochs=1
    )
    print(time.time() - start)
    print(evaluate(
        model=model, params=params, inputs=test_images, targets=test_labels, batch_size=100
    ))


if __name__ == '__main__':
    main()
