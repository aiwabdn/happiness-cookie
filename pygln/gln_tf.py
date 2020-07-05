import scipy
import tensorflow as tf
from typing import Sequence


class OnlineUpdateModule(tf.Module):

    def __init__(
        self, learning_rate: float, pred_clipping: float, weight_clipping: float, name: str = None
    ):
        assert learning_rate > 0.0
        assert 0.0 < pred_clipping < 1.0
        assert weight_clipping is None or weight_clipping >= 1.0

        self.learning_rate = learning_rate
        self.pred_clipping = pred_clipping
        self.weight_clipping = weight_clipping

    def predict(self, logits, context, target=None):
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
        self.context_map_size = context_map_size
        self.classes = classes

        logits_size = input_size + int(bias)
        num_context_indices = 1 << self.context_map_size
        if self.classes is None:
            context_maps_shape = (1, self.size, self.context_map_size, context_size)
            context_bias_shape = (self.size, self.context_map_size)
            weights_shape = (self.size, num_context_indices, logits_size)
            bias_shape = (1, 1)
        else:
            context_maps_shape = (1, self.classes, self.size, self.context_map_size, context_size)
            context_bias_shape = (1, self.classes, self.size, self.context_map_size)
            weights_shape = (self.classes, self.size, num_context_indices, logits_size)
            bias_shape = (1, self.classes, 1)

        initializer = tf.constant_initializer(value=(1.0 / logits_size))(shape=weights_shape)
        self.weights = tf.Variable(
            initial_value=initializer, trainable=True, name='weights', dtype=tf.dtypes.float32
        )
        if bias:
            initializer = tf.random_uniform_initializer(
                minval=scipy.special.logit(self.pred_clipping),
                maxval=scipy.special.logit(1.0 - self.pred_clipping)
            )(shape=bias_shape)
            self.bias = tf.Variable(
                initial_value=initializer, trainable=False, name='bias', dtype=tf.dtypes.float32
            )
        else:
            self.bias = None

        if context_bias:
            context_maps = tf.random.normal(shape=context_maps_shape, dtype=tf.dtypes.float32)
            norm = tf.norm(context_maps, axis=-1, keepdims=True)
            self.context_maps = tf.Variable(
                initial_value=(context_maps / norm), trainable=False, name='context_maps',
                dtype=tf.dtypes.float32
            )
            initializer = tf.random_normal_initializer()(shape=context_bias_shape)
            self.context_bias = tf.Variable(
                initial_value=initializer, trainable=False, name='context_bias',
                dtype=tf.dtypes.float32
            )
        else:
            initializer = tf.random_normal_initializer()(shape=context_maps_shape)
            self.context_maps = tf.Variable(
                initial_value=initializer, trainable=False, name='context_maps',
                dtype=tf.dtypes.float32
            )
            self.context_bias = None

    def predict(self, logits, context, target=None):
        context = tf.expand_dims(tf.expand_dims(context, axis=1), axis=1)
        if self.classes is not None:
            context = tf.expand_dims(context, axis=1)

        if self.context_bias is None:
            context_bias = 0.0
        else:
            context_bias = self.context_bias
        context_index = tf.math.reduce_sum(self.context_maps * context, axis=-1) > context_bias

        if self.classes is None:
            context_map_values = tf.constant([[[1 << n for n in range(self.context_map_size)]]])
        else:
            context_map_values = tf.constant([[[[1 << n for n in range(self.context_map_size)]]]])
        context_index = tf.where(context_index, context_map_values, 0)
        context_index = tf.math.reduce_sum(context_index, axis=-1, keepdims=True)

        batch_size = tf.shape(logits)[0]
        if self.classes is None:
            neuron_index = tf.constant([[[n] for n in range(self.size)]])
            neuron_index = tf.tile(neuron_index, multiples=(batch_size, 1, 1))
            context_index = tf.concat([neuron_index, context_index], axis=-1)
        else:
            class_neuron_index = tf.constant(
                [[[[c, n] for n in range(self.size)] for c in range(self.classes)]]
            )
            class_neuron_index = tf.tile(class_neuron_index, multiples=(batch_size, 1, 1, 1))
            context_index = tf.concat([class_neuron_index, context_index], axis=-1)
        weights = tf.gather_nd(self.weights, indices=context_index)

        if self.classes is None:
            bias = tf.tile(self.bias, multiples=(batch_size, 1))
        else:
            bias = tf.tile(self.bias, multiples=(batch_size, 1, 1))
        logits = tf.concat([logits, bias], axis=-1)
        logits = tf.expand_dims(logits, axis=-1)

        output_logits = tf.linalg.matmul(weights, logits)
        output_logits = tf.clip_by_value(
            output_logits, clip_value_min=scipy.special.logit(self.pred_clipping),
            clip_value_max=scipy.special.logit(1.0 - self.pred_clipping)
        )

        if target is not None:
            logits = tf.expand_dims(tf.squeeze(logits, axis=-1), axis=-2)
            output_preds = tf.math.sigmoid(output_logits)
            target = tf.expand_dims(tf.expand_dims(target, axis=-1), axis=-1)
            delta = self.learning_rate * (target - output_preds) * logits

            if self.weight_clipping is None:
                self.weights.scatter_nd_add(indices=context_index, updates=delta)
            else:
                weights = tf.clip_by_value(
                    weights + delta, clip_value_min=-self.weight_clipping,
                    clip_value_max=self.weight_clipping
                )
                self.weights.scatter_nd_update(indices=context_index, updates=weights)

        return tf.squeeze(output_logits, axis=-1)


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

        # Base predictions
        if self.base_preds is None:
            self.base_logits = None
        else:
            if self.classes is None:
                base_logits_shape = (1, self.base_preds)
            else:
                base_logits_shape = (1, self.classes, self.base_preds)
            initializer = tf.random_uniform_initializer(
                minval=scipy.special.logit(self.pred_clipping),
                maxval=scipy.special.logit(1.0 - self.pred_clipping)
            )(shape=base_logits_shape)
            self.base_logits = tf.Variable(
                initial_value=initializer, trainable=False, name='base_logits',
                dtype=tf.dtypes.float32
            )

        # TF-compiled predict function
        self._tf_predict = tf.function(
            func=(lambda input: self._predict(input)),
            input_signature=[tf.TensorSpec(shape=(None, input_size), dtype=tf.dtypes.float32)],
            autograph=False
        )

        # TF-compiled update function
        if self.classes is None:
            self.target_dtype = tf.dtypes.bool
        else:
            self.target_dtype = tf.dtypes.int64
        self._tf_update = tf.function(
            func=(lambda input, target: self._predict(input, target)),
            input_signature=[
                tf.TensorSpec(shape=(None, input_size), dtype=tf.dtypes.float32),
                tf.TensorSpec(shape=(None,), dtype=self.target_dtype)
            ], autograph=False
        )

    def predict(self, input, target=None):
        input = tf.convert_to_tensor(input, dtype=tf.dtypes.float32)
        if target is None:  # predict
            return self._tf_predict(input=input).numpy()
        else:  # predict with online update
            target = tf.convert_to_tensor(target, dtype=self.target_dtype)
            return self._tf_update(input=input, target=target).numpy()

    def _predict(self, input, target=None):
        # Base predictions
        if self.base_logits is None:
            logits = tf.clip_by_value(
                input, clip_value_min=self.pred_clipping, clip_value_max=(1.0 - self.pred_clipping)
            )
            logits = tf.math.log(logits / (1.0 - logits))
            if self.classes is not None:
                logits = tf.expand_dims(logits, axis=1)
                logits = tf.tile(logits, multiples=(1, self.classes, 1))
        else:
            batch_size = tf.shape(input)[0]
            if self.classes is None:
                logits = tf.tile(self.base_logits, multiples=(batch_size, 1))
            else:
                logits = tf.tile(self.base_logits, multiples=(batch_size, 1, 1))

        # Turn class integer into one-hot
        if target is not None:
            if self.classes is None:
                target = tf.where(target, 1.0, 0.0)
            else:
                target = tf.one_hot(target, depth=self.classes)

        # Layers
        for n, layer in enumerate(self.layers):
            logits = layer.predict(logits=logits, context=input, target=target)

        # Output prediction
        logits = tf.squeeze(logits, axis=-1)
        if self.classes is None:
            return logits > 0.0
        else:
            return tf.math.argmax(logits, axis=1)

    def evaluate(self, inputs, targets, batch_size):
        assert inputs.shape[0] % batch_size == 0

        inputs = tf.convert_to_tensor(inputs, dtype=tf.dtypes.float32)
        targets = tf.convert_to_tensor(targets, dtype=self.target_dtype)
        num_instances = inputs.shape[0]

        @tf.function
        def body(n, num_correct):
            batch = tf.range(n * batch_size, (n + 1) * batch_size) % num_instances
            prediction = self._tf_predict(input=tf.gather(inputs, batch))
            num_correct += tf.math.count_nonzero(prediction == tf.gather(targets, batch))
            return n + 1, num_correct

        @tf.function
        def cond(n, accuracy):
            return True

        num_iterations = num_instances // batch_size
        _, num_correct = tf.while_loop(
            cond=cond, body=body, loop_vars=(0, 0), maximum_iterations=num_iterations
        )
        return num_correct.numpy() / num_instances

    def train(self, inputs, targets, batch_size, num_iterations=None, num_epochs=None):
        assert (num_iterations is None) is not (num_epochs is None)
        assert inputs.shape[0] % batch_size == 0

        inputs = tf.convert_to_tensor(inputs, dtype=tf.dtypes.float32)
        targets = tf.convert_to_tensor(targets, dtype=self.target_dtype)
        num_instances = inputs.shape[0]

        @tf.function
        def body(n):
            if num_epochs is None:
                batch = tf.random.uniform(batch_size, maxval=num_instances, dtype=tf.dtypes.int64)
            else:
                batch = tf.range(n * batch_size, (n + 1) * batch_size) % num_instances
            self._tf_update(input=tf.gather(inputs, batch), target=tf.gather(targets, batch))
            return (n + 1,)

        @tf.function
        def cond(n):
            return True

        if num_epochs is not None:
            num_iterations = num_instances // batch_size
        n, = tf.while_loop(cond=cond, body=body, loop_vars=(0,), maximum_iterations=num_iterations)
        assert n.numpy().item() == num_iterations, (n, num_iterations)


def main():
    import time
    import datasets

    train_images, train_labels, test_images, test_labels = datasets.get_mnist()

    model = GLN(
        layer_sizes=[32, 32, 1], input_size=train_images.shape[1], context_map_size=4,
        learning_rate=3e-5, pred_clipping=0.001, weight_clipping=5.0, classes=10, base_preds=None
    )

    print('Accuracy:', model.evaluate(test_images, test_labels, batch_size=100))

    start = time.time()
    model.train(train_images, train_labels, batch_size=1, num_epochs=1)
    print('Time:', time.time() - start)

    print('Accuracy:', model.evaluate(test_images, test_labels, batch_size=100))


if __name__ == '__main__':
    main()
