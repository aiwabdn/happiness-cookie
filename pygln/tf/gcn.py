import numpy as np
import scipy.special
import tensorflow as tf
from typing import Callable, Optional, Sequence, Tuple, Union

# from ..base import GLNBase
# from .gln import OnlineUpdateModule


class Conv2d(tf.Module):  # OnlineUpdateModule

    def __init__(self, size: int, input_size: int, window_size: int,
                 context_size: int, context_map_size: int, num_classes: int,
                 learning_rate: float, pred_clipping: float, weight_clipping: float,
                 bias: bool, context_bias: bool):
        # super().__init__(learning_rate, pred_clipping, weight_clipping)
        assert isinstance(learning_rate, float)
        assert 0.0 < pred_clipping < 1.0
        assert weight_clipping >= 1.0
        self.learning_rate = learning_rate
        self.pred_clipping = pred_clipping
        self.weight_clipping = weight_clipping

        assert size > 0 and input_size > 0 and context_size > 0
        assert context_map_size >= 1
        assert num_classes >= 2

        self.size = size
        self.input_size = input_size
        self.window_size = window_size
        self.context_size = context_size
        self.context_map_size = context_map_size
        self.num_classes = num_classes if num_classes > 2 else 1

        input_window_size = self.input_size * self.window_size * self.window_size + int(bias)
        num_context_indices = 1 << self.context_map_size
        weights_shape = (self.num_classes, self.size, num_context_indices, input_window_size)
        initializer = tf.constant_initializer(value=(1.0 / input_window_size))(shape=weights_shape)
        self.weights = tf.Variable(
            initial_value=initializer, trainable=True, name='weights', dtype=tf.dtypes.float32
        )

        if bias:
            bias_shape = (1, self.num_classes, 1, 1, 1)
            initializer = tf.random_uniform_initializer(
                minval=scipy.special.logit(self.pred_clipping),
                maxval=scipy.special.logit(1.0 - self.pred_clipping)
            )(shape=bias_shape)
            self.bias = tf.Variable(
                initial_value=initializer, trainable=False, name='bias', dtype=tf.dtypes.float32
            )
        else:
            self.bias = None

        context_window_size = self.context_size * self.window_size * self.window_size
        context_maps_shape = (
            1, self.num_classes, self.size, 1, 1, self.context_map_size, context_window_size
        )
        if context_bias:
            context_maps = tf.random.normal(shape=context_maps_shape, dtype=tf.dtypes.float32)
            norm = tf.norm(context_maps, axis=-1, keepdims=True)
            self.context_maps = tf.Variable(
                initial_value=(context_maps / norm), trainable=False, name='context_maps',
                dtype=tf.dtypes.float32
            )

            context_bias_shape = (1, self.num_classes, self.size, 1, 1, self.context_map_size)
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
            self.context_bias = 0.0

    def predict(self, logits, context, target=None):
        context = tf.expand_dims(context, axis=1)
        context = tf.extract_volume_patches(
            context, ksizes=(1, 1, self.window_size, self.window_size, 1),
            strides=(1, 1, 1, 1, 1), padding='SAME'
        )
        shape = tf.shape(context)
        context = tf.reshape(context, shape=(shape[0], 1, shape[2], shape[3], self.context_size * self.window_size * self.window_size))
        context = tf.expand_dims(tf.expand_dims(context, axis=1), axis=-2)
        context_index = tf.math.reduce_sum(self.context_maps * context, axis=-1) > self.context_bias

        context_map_values = tf.constant([[[[[[1 << n for n in range(self.context_map_size)]]]]]])
        context_index = tf.where(context_index, context_map_values, 0)
        context_index = tf.math.reduce_sum(context_index, axis=-1, keepdims=True)

        batch_size = tf.shape(logits)[0]
        class_neuron_index = tf.constant(
            [[[[[[c, n]]] for n in range(self.size)] for c in range(self.num_classes)]]
        )
        multiples = tf.concat([(batch_size, 1, 1), tf.shape(context_index)[3:5], (1,)], axis=0)
        class_neuron_index = tf.tile(class_neuron_index, multiples=multiples)
        context_index = tf.concat([class_neuron_index, context_index], axis=-1)

        weights = tf.gather_nd(self.weights, indices=context_index)

        logits = tf.extract_volume_patches(
            logits, ksizes=(1, 1, self.window_size, self.window_size, 1),
            strides=(1, 1, 1, 1, 1), padding='SAME'
        )
        shape = tf.shape(logits)
        logits = tf.reshape(logits, shape=(shape[0], 1, shape[2], shape[3], self.input_size * self.window_size * self.window_size))
        # logits = tf.expand_dims(logits, axis=1)
        if self.bias is not None:
            multiples = tf.concat([(batch_size, 1), tf.shape(context_index)[3:5], (1,)], axis=0)
            bias = tf.tile(self.bias, multiples=multiples)
            logits = tf.concat([logits, bias], axis=-1)
        logits = tf.expand_dims(logits, axis=-1)

        output_logits = tf.linalg.matmul(tf.transpose(weights, perm=(0, 1, 3, 4, 2, 5)), logits)
        output_logits = tf.clip_by_value(
            output_logits,
            clip_value_min=scipy.special.logit(self.pred_clipping),
            clip_value_max=scipy.special.logit(1.0 - self.pred_clipping)
        )

        if target is None:
            return tf.squeeze(output_logits, axis=-1)

        else:
            logits = tf.expand_dims(tf.squeeze(logits, axis=-1), axis=-2)
            output_preds = tf.math.sigmoid(output_logits)
            target = tf.expand_dims(tf.expand_dims(target, axis=-1), axis=-1)
            delta = self.learning_rate * (target - output_preds) * logits  # self.learning_rate.value()
            delta = tf.transpose(delta, perm=(0, 1, 4, 2, 3, 5))

            if self.weight_clipping is None:
                assignment = self.weights.scatter_nd_add(indices=context_index, updates=delta)
            else:
                weights = tf.clip_by_value(
                    weights + delta,
                    clip_value_min=-self.weight_clipping,
                    clip_value_max=self.weight_clipping
                )
                assignment = self.weights.scatter_nd_update(indices=context_index, updates=weights)

            with tf.control_dependencies(control_inputs=(assignment,)):
                return tf.squeeze(output_logits, axis=-1)


class GCN(tf.Module):  # GLNBase
    """
    TensorFlow implementation of Gated Linear Networks (https://arxiv.org/abs/1910.01526).

    Args:
        layer_sizes (list[int >= 1]): List of layer output sizes.
        input_size (int >= 1): Input vector size.
        num_classes (int >= 2): For values >2, turns GLN into a multi-class classifier by internally
            creating a one-vs-all binary GLN classifier per class and return the argmax as output.
        context_map_size (int >= 1): Context dimension, i.e. number of context halfspaces.
        bias (bool): Whether to add a bias prediction in each layer.
        context_bias (bool): Whether to use a random non-zero bias for context halfspace gating.
        base_predictor (np.array[N] -> np.array[K]): If given, maps the N-dim input vector to a
            corresponding K-dim vector of base predictions (could be a constant prior), instead of
            simply using the clipped input vector itself.
        learning_rate (float > 0.0): Update learning rate.
        pred_clipping (0.0 < float < 0.5): Clip predictions into [p, 1 - p] at each layer.
        weight_clipping (float > 0.0): Clip weights into [-w, w] after each update.
    """
    def __init__(self,
                 layer_sizes: Sequence[int],
                 input_size: int,
                 window_size: int,
                 num_classes: int = 2,
                 context_map_size: int = 4,
                 bias: bool = True,
                 context_bias: bool = True,
                 base_predictor: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                 learning_rate: float = 1e-4,
                 pred_clipping: float = 1e-3,
                 weight_clipping: float = 5.0):

        tf.Module.__init__(self, name='GLN')

        # GLNBase.__init__(self, layer_sizes, input_size, num_classes,
        #                  context_map_size, bias, context_bias, base_predictor,
        #                  learning_rate, pred_clipping, weight_clipping)
        assert len(layer_sizes) > 0 and layer_sizes[-1] == 1
        self.layer_sizes = tuple(layer_sizes)
        assert input_size > 0
        self.input_size = input_size
        assert num_classes >= 2
        self.num_classes = num_classes
        assert context_map_size >= 1
        self.context_map_size = context_map_size
        self.bias = bias
        self.context_bias = context_bias
        if base_predictor is None:
            # self.base_predictor = (
            #     lambda x: (x - x.min(axis=1, keepdims=True)) /
            #     (x.max(axis=1, keepdims=True) - x.min(axis=1, keepdims=True))
            # )
            self.base_predictor = (lambda x: x)
            self.base_pred_size = self.input_size
        else:
            self.base_predictor = base_predictor
            dummy_input = np.zeros(shape=(1, self.input_size))
            dummy_pred = self.base_predictor(dummy_input)
            assert dummy_pred.dtype in (np.float32, np.float64)
            assert dummy_pred.ndim == 2 and dummy_pred.shape[0] == 1
            self.base_pred_size = dummy_pred.shape[1]
        if isinstance(learning_rate, float):
            assert learning_rate > 0.0
        self.learning_rate = learning_rate
        assert 0.0 < pred_clipping < 1.0
        self.pred_clipping = pred_clipping
        assert weight_clipping > 0.0
        self.weight_clipping = weight_clipping

        # Initialize layers
        self.layers = list()
        previous_size = self.base_pred_size
        for size in self.layer_sizes:
            self.layers.append(
                Conv2d(size=size,
                       input_size=previous_size,
                       window_size=window_size,
                       context_size=self.input_size,
                       context_map_size=self.context_map_size,
                       num_classes=self.num_classes,
                       learning_rate=self.learning_rate,
                       pred_clipping=self.pred_clipping,
                       weight_clipping=self.weight_clipping,
                       bias=self.bias,
                       context_bias=self.context_bias))
            previous_size = size

        # TF-compiled predict function
        self._tf_predict = tf.function(
            func=self._predict,
            input_signature=[
                tf.TensorSpec(shape=((None, None, None) + (self.base_pred_size,)),
                              dtype=tf.dtypes.float32),
                tf.TensorSpec(shape=((None, None, None) + (self.input_size,)),
                              dtype=tf.dtypes.float32)
            ], autograph=False
        )

        # TF-compiled update function
        self.target_dtype = tf.dtypes.int64
        self._tf_update = tf.function(
            func=self._predict,
            input_signature=[
                tf.TensorSpec(shape=((None, None, None) + (self.base_pred_size,)),
                              dtype=tf.dtypes.float32),
                tf.TensorSpec(shape=((None, None, None) + (self.input_size,)),
                              dtype=tf.dtypes.float32),
                tf.TensorSpec(shape=(None, None, None), dtype=self.target_dtype)
                # tf.TensorSpec(shape=(None,), dtype=self.target_dtype)
            ], autograph=False
        )

    def predict(
        self, input: np.ndarray, target: Optional[np.ndarray] = None, return_probs: bool = False
    ) -> np.ndarray:
        """
        Predict the class for the given inputs, and optionally update the weights.

        Args:
            input (np.array[B, N]): Batch of B N-dim float input vectors.
            target (np.array[B]): Optional batch of B target class labels (bool, or int if
                num_classes given) which, if given, triggers an online update if given.
            return_probs (bool): Whether to return the classification probability (for each
                one-vs-all classifier if num_classes given) instead of the class.

        Returns:
            Predicted class per input instance (bool, or int if num_classes given),
            or classification probabilities if return_probs set.
        """

        # Base predictions
        base_preds = self.base_predictor(input)
        base_preds = tf.convert_to_tensor(base_preds, dtype=tf.dtypes.float32)

        # Context
        context = tf.convert_to_tensor(input, dtype=tf.dtypes.float32)

        if target is None:
            # Predict without update
            logits = self._tf_predict(base_preds=base_preds, context=context)

        else:
            # Target
            target = tf.convert_to_tensor(target, dtype=self.target_dtype)

            # Predict with update
            logits = self._tf_update(base_preds=base_preds,
                                     context=context,
                                     target=target)

        if self.num_classes == 2:
            logits = np.squeeze(logits, axis=1)

        if return_probs:
            return scipy.special.expit(logits)
        elif self.num_classes == 2:
            return logits > 0.0
        else:
            return np.argmax(logits, axis=1)

    def _predict(self, base_preds, context, target=None):
        # Base logits
        base_preds = tf.clip_by_value(
            base_preds, clip_value_min=self.pred_clipping, clip_value_max=(1.0 - self.pred_clipping)
        )
        logits = tf.math.log(base_preds / (1.0 - base_preds))
        logits = tf.expand_dims(logits, axis=1)
        logits = tf.tile(
            logits, multiples=(1, self.num_classes if self.num_classes > 2 else 1, 1, 1, 1)
        )

        # Turn target class into one-hot
        if target is not None:
            target = tf.one_hot(target, depth=self.num_classes, axis=1)
            if self.num_classes == 2:
                target = target[:, 1:]

        # Layers
        for layer in self.layers:
            logits = layer.predict(logits=logits, context=context, target=target)

        return tf.squeeze(logits, axis=-1)


def main():
    from pygln import utils

    X_train, y_train, X_test, y_test = utils.get_mnist()

    model = GCN(layer_sizes=[16, 16, 1], input_size=1, window_size=3)
    X_train = np.reshape(X_train, (X_train.shape[0], 28, 28, 1))
    X_test = np.reshape(X_test, (X_test.shape[0], 28, 28, 1))
    y_train = np.reshape(X_train > 0.5, (y_train.shape[0], 28, 28))
    y_test = np.reshape(X_test > 0.5, (y_test.shape[0], 28, 28))
    for n in range(X_train.shape[0]):
        model.predict(X_train[n: n + 1], y_train[n: n + 1])
    num_correct = 0
    for n in range(X_test.shape[0]):
        prediction = model.predict(X_test[n: n + 1])
        num_correct += np.count_nonzero(prediction == y_test[n: n + 1])
    print(num_correct / y_test.size)


if __name__ == '__main__':
    main()
