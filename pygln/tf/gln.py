import numpy as np
import scipy.special
import tensorflow as tf
from typing import Callable, Optional, Sequence

from pygln.base import GLNBase


class DynamicParameter(tf.Module):
    def __init__(self, name: str):
        super().__init__(name=name)

        self.step = tf.Variable(initial_value=0.0,
                                trainable=False,
                                name='step',
                                dtype=tf.dtypes.float32)

    def value(self):
        return self.step.assign_add(1.0)


class ConstantParameter(DynamicParameter):
    def __init__(self, constant_value, name: str):
        DynamicParameter.__init__(self, name)

        assert isinstance(constant_value, float)
        self.constant_value = constant_value

    def value(self):
        return tf.constant(self.constant_value)


class PaperLearningRate(DynamicParameter):
    def value(self):
        return tf.math.minimum(100.0 / super().value(), 0.01)


class OnlineUpdateModule(tf.Module):
    def __init__(self,
                 learning_rate: DynamicParameter,
                 pred_clipping: float,
                 weight_clipping: float,
                 name: str = None):
        assert isinstance(learning_rate, DynamicParameter)
        assert 0.0 < pred_clipping < 1.0
        assert weight_clipping >= 1.0

        self.learning_rate = learning_rate
        self.pred_clipping = pred_clipping
        self.weight_clipping = weight_clipping

    def predict(self, logits, context, target=None):
        raise NotImplementedError()


class Linear(OnlineUpdateModule):
    def __init__(self, size: int, input_size: int, context_size: int,
                 context_map_size: int, num_classes: int,
                 learning_rate: DynamicParameter, pred_clipping: float,
                 weight_clipping: float, bias: bool, context_bias: bool):
        super().__init__(learning_rate, pred_clipping, weight_clipping)

        assert size > 0 and input_size > 0 and context_size > 0
        assert context_map_size >= 2
        assert num_classes >= 1

        self.size = size
        self.context_map_size = context_map_size
        self.num_classes = num_classes

        logits_size = input_size + int(bias)
        num_context_indices = 1 << self.context_map_size
        weights_shape = (self.num_classes, self.size, num_context_indices,
                         logits_size)
        initializer = tf.constant_initializer(value=(1.0 / logits_size))(
            shape=weights_shape)
        self.weights = tf.Variable(initial_value=initializer,
                                   trainable=True,
                                   name='weights',
                                   dtype=tf.dtypes.float32)

        if bias:
            bias_shape = (1, self.num_classes, 1)
            initializer = tf.random_uniform_initializer(
                minval=scipy.special.logit(self.pred_clipping),
                maxval=scipy.special.logit(1.0 - self.pred_clipping))(
                    shape=bias_shape)
            self.bias = tf.Variable(initial_value=initializer,
                                    trainable=False,
                                    name='bias',
                                    dtype=tf.dtypes.float32)
        else:
            self.bias = None

        context_maps_shape = (1, self.num_classes, self.size,
                              self.context_map_size, context_size)
        if context_bias:
            context_maps = tf.random.normal(shape=context_maps_shape,
                                            dtype=tf.dtypes.float32)
            norm = tf.norm(context_maps, axis=-1, keepdims=True)
            self.context_maps = tf.Variable(initial_value=(context_maps /
                                                           norm),
                                            trainable=False,
                                            name='context_maps',
                                            dtype=tf.dtypes.float32)

            context_bias_shape = (1, self.num_classes, self.size,
                                  self.context_map_size)
            initializer = tf.random_normal_initializer()(
                shape=context_bias_shape)
            self.context_bias = tf.Variable(initial_value=initializer,
                                            trainable=False,
                                            name='context_bias',
                                            dtype=tf.dtypes.float32)

        else:
            initializer = tf.random_normal_initializer()(
                shape=context_maps_shape)
            self.context_maps = tf.Variable(initial_value=initializer,
                                            trainable=False,
                                            name='context_maps',
                                            dtype=tf.dtypes.float32)
            self.context_bias = 0.0

    def predict(self, logits, context, target=None):
        context = tf.expand_dims(tf.expand_dims(tf.expand_dims(context,
                                                               axis=1),
                                                axis=1),
                                 axis=1)
        context_index = tf.math.reduce_sum(self.context_maps * context,
                                           axis=-1) > self.context_bias

        context_map_values = tf.constant(
            [[[[1 << n for n in range(self.context_map_size)]]]])
        context_index = tf.where(context_index, context_map_values, 0)
        context_index = tf.math.reduce_sum(context_index,
                                           axis=-1,
                                           keepdims=True)

        batch_size = tf.shape(logits)[0]
        class_neuron_index = tf.constant([[[[c, n] for n in range(self.size)]
                                           for c in range(self.num_classes)]])
        class_neuron_index = tf.tile(class_neuron_index,
                                     multiples=(batch_size, 1, 1, 1))
        context_index = tf.concat([class_neuron_index, context_index], axis=-1)

        weights = tf.gather_nd(self.weights, indices=context_index)

        bias = tf.tile(self.bias, multiples=(batch_size, 1, 1))
        logits = tf.concat([logits, bias], axis=-1)
        logits = tf.expand_dims(logits, axis=-1)

        output_logits = tf.linalg.matmul(weights, logits)
        output_logits = tf.clip_by_value(
            output_logits,
            clip_value_min=scipy.special.logit(self.pred_clipping),
            clip_value_max=scipy.special.logit(1.0 - self.pred_clipping))

        if target is None:
            return tf.squeeze(output_logits, axis=-1)

        else:
            logits = tf.expand_dims(tf.squeeze(logits, axis=-1), axis=-2)
            output_preds = tf.math.sigmoid(output_logits)
            target = tf.expand_dims(tf.expand_dims(target, axis=-1), axis=-1)
            delta = self.learning_rate.value() * (target -
                                                  output_preds) * logits

            if self.weight_clipping is None:
                assignment = self.weights.scatter_nd_add(indices=context_index,
                                                         updates=delta)
            else:
                weights = tf.clip_by_value(
                    weights + delta,
                    clip_value_min=-self.weight_clipping,
                    clip_value_max=self.weight_clipping)
                assignment = self.weights.scatter_nd_update(
                    indices=context_index, updates=weights)

            with tf.control_dependencies(control_inputs=(assignment, )):
                return tf.squeeze(output_logits, axis=-1)


class GLN(tf.Module, GLNBase):
    """
    TensorFlow implementation of Gated Linear Networks (https://arxiv.org/abs/1910.01526).

    Args:
        layer_sizes (list[int >= 1]): List of layer output sizes.
        input_size (int >= 1): Input vector size.
        context_map_size (int >= 1): Context dimension, i.e. number of context halfspaces.
        num_classes (int >= 2): If given, turns GLN into a multi-class classifier by internally
            creating N one-vs-all binary GLN classifiers and return the argmax as output class.
        base_predictor (np.array[n] -> np.array[k]): If given, maps the n-dim input vector to a
            corresponding k-dim vector of base predictions (could be a constant prior), instead of
            simply using the clipped input vector itself.
        learning_rate (float > 0.0): Update learning rate.
        pred_clipping (0.0 < float < 0.5): Clip predictions into [p, 1 - p] at each layer.
        weight_clipping (float > 0.0): Clip weights into [-w, w] after each update.
        bias (bool): Whether to add a bias prediction in each layer.
        context_bias (bool): Whether to use a random non-zero bias for context halfspace gating.
    """

    def __init__(self,
                 layer_sizes: Sequence[int],
                 input_size: int,
                 context_map_size: int = 4,
                 num_classes: Optional[int] = None,
                 base_predictor: Optional[
                     Callable[[np.ndarray], np.ndarray]] = None,
                 learning_rate: float = 1e-4,
                 pred_clipping: float = 1e-3,
                 weight_clipping: float = 5.0,
                 bias: bool = True,
                 context_bias: bool = True):

        tf.Module.__init__(self, name='GLN')
        GLNBase.__init__(self, layer_sizes, input_size, context_map_size,
                         num_classes, base_predictor, learning_rate,
                         pred_clipping, weight_clipping, bias, context_bias)

        # Learning rate as dynamic parameter
        if self.learning_rate == 'paper':
            self.learning_rate = PaperLearningRate(name='learning_rate')
        else:
            self.learning_rate = ConstantParameter(self.learning_rate,
                                                   name='learning_rate')

        # Initialize layers
        self.layers = list()
        previous_size = self.base_pred_size
        for size in self.layer_sizes:
            self.layers.append(
                Linear(size=size,
                       input_size=previous_size,
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
                tf.TensorSpec(shape=(None, self.base_pred_size),
                              dtype=tf.dtypes.float32),
                tf.TensorSpec(shape=(None, self.input_size),
                              dtype=tf.dtypes.float32)
            ],
            autograph=False)

        # TF-compiled update function
        if self.num_classes == 1:
            self.target_dtype = tf.dtypes.bool
        else:
            self.target_dtype = tf.dtypes.int64
        self._tf_update = tf.function(
            func=self._predict,
            input_signature=[
                tf.TensorSpec(shape=(None, self.base_pred_size),
                              dtype=tf.dtypes.float32),
                tf.TensorSpec(shape=(None, self.input_size),
                              dtype=tf.dtypes.float32),
                tf.TensorSpec(shape=(None, ), dtype=self.target_dtype)
            ],
            autograph=False)

    def predict(self, input: np.ndarray, target: np.ndarray = None, return_probs: bool = False):
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
            if self.num_classes == 1:
                target = tf.convert_to_tensor(target, dtype=self.target_dtype)
            else:
                target = tf.convert_to_tensor(target, dtype=self.target_dtype)

            # Predict with update
            logits = self._tf_update(base_preds=base_preds, context=context, target=target)

        if return_probs:
            return scipy.special.expit(logits)
        elif self.num_classes == 1:
            return np.squeeze(logits, axis=-1) > 0.0
        else:
            return np.argmax(logits, axis=1)

    def _predict(self, base_preds, context, target=None):
        # Base logits
        base_preds = tf.clip_by_value(base_preds,
                                      clip_value_min=self.pred_clipping,
                                      clip_value_max=(1.0 -
                                                      self.pred_clipping))
        logits = tf.math.log(base_preds / (1.0 - base_preds))
        logits = tf.expand_dims(logits, axis=1)
        logits = tf.tile(logits, multiples=(1, self.num_classes, 1))

        # Turn target class into one-hot
        if target is not None:
            if self.num_classes == 1:
                target = tf.expand_dims(tf.where(target, 1.0, 0.0), axis=1)
            else:
                target = tf.one_hot(target, depth=self.num_classes)

        # Layers
        for layer in self.layers:
            logits = layer.predict(logits=logits,
                                   context=context,
                                   target=target)
        return tf.squeeze(logits, axis=-1)
