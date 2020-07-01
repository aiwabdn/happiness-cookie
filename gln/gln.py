# %%
import numpy as np

from test_mnist import get_mnist_metrics

np.set_printoptions(precision=4, suppress=True)


# %%
def sigmoid(X):
    return 1 / (1 + np.exp(-X))


class Neuron():
    def __init__(self,
                 input_dim=128,
                 side_info_dim=784,
                 context_dim=4,
                 mu=0.0,
                 std=0.1):
        self.v = np.random.normal(loc=mu,
                                  scale=std,
                                  size=(side_info_dim, context_dim))
        self.v = self.v / np.linalg.norm(self.v, ord=2, axis=0, keepdims=True)
        self.b = np.random.normal(size=(context_dim, 1))
        self.weights = np.ones(shape=(input_dim,
                                      2**context_dim)) * (1 / input_dim)
        self.boolean_converter = np.array([[2**i] for i in range(context_dim)])

    def forward(self, logit_previous, side_information):
        projection = self.v.T.dot(side_information)
        if projection.ndim == 1:
            projection = projection.reshape(-1, 1)
        self.current_contexts = self.boolean_converter.T.dot(
            (projection > self.b).astype(np.int)).flatten()
        self.current_selected_weights = self.weights[:, self.current_contexts]
        self.output_logits = np.multiply(self.current_selected_weights,
                                         logit_previous).sum(axis=0)
        self.logit_previous = logit_previous
        return self.output_logits

    def update(self, targets, learning_rate=0.001):
        sigmoids = sigmoid(self.output_logits)
        update_value = learning_rate * (sigmoids -
                                        targets) * self.logit_previous
        for i in range(update_value.shape[-1]):
            self.weights[:, self.current_contexts[i]] -= update_value[:, i]


class Layer():
    def __init__(self,
                 num_neurons=128,
                 input_dim=128,
                 side_info_dim=128,
                 epsilon=0.05):
        self.neurons = [
            Neuron(input_dim, side_info_dim)
            for _ in range(max(1, num_neurons - 1))
        ]
        self.bias = np.random.uniform(epsilon, 1 - epsilon)

    def forward(self, logit_previous, side_information):
        output_logits = []
        if len(self.neurons) > 1:
            # no bias for the output neuron
            output_logits.append(np.repeat(self.bias,
                                           logit_previous.shape[-1]))
        for n in self.neurons:
            output_logits.append(n.forward(logit_previous, side_information))
        output = np.vstack(output_logits)
        return output

    def update(self, targets, learning_rate=0.001):
        for n in self.neurons:
            n.update(targets)


class LayerVec():
    def __init__(self,
                 num_neurons=128,
                 input_dim=128,
                 side_info_dim=784,
                 context_dim=4,
                 mu=0.0,
                 std=0.1,
                 epsilon=0.05,
                 beta=1.5):

        self.num_neurons = num_neurons
        # constant bias for the layer
        self.bias = np.random.uniform(epsilon, 1 - epsilon)
        # context function for halfspace gating
        self.v = np.random.normal(loc=mu,
                                  scale=std,
                                  size=(num_neurons, context_dim,
                                        side_info_dim))
        self.v /= np.linalg.norm(self.v, axis=2, keepdims=True)
        # constant values for halfspace gating
        self.b = np.random.normal(loc=mu,
                                  scale=std,
                                  size=(num_neurons, context_dim, 1))
        # array to convert binary context to index
        self.boolean_converter = np.array([[2**i] for i in range(context_dim)])
        # weights for the whole layer
        self.weights = np.ones(
            (num_neurons, 2**context_dim, input_dim)) * (1 / input_dim)
        # clipping value for outputs of neurons
        self.epsilon = epsilon
        # clipping value for weights of layer
        self.beta = beta

    def forward(self, logit_previous, side_information):
        # project side information and determine context index
        projection = np.matmul(self.v, side_information)
        binary = (projection > self.b).astype(np.int)
        self.current_contexts = np.squeeze(
            np.sum(binary * self.boolean_converter, axis=1))

        # select all contexts across all neurons in layer
        self.current_selected_weights = self.weights[np.arange(
            self.num_neurons).reshape(-1, 1), self.current_contexts, :]

        # compute logit output
        # matmul duplicates results, so take diagonal
        self.output_logits = np.matmul(self.current_selected_weights,
                                       logit_previous).diagonal(axis1=1,
                                                                axis2=2)

        # if not final output layer
        if self.num_neurons > 1:
            # make array writeable
            self.output_logits.setflags(write=1)
            # assign output of first neuron to bias
            # done for ease of computation
            self.output_logits[0] = self.bias

        # save the previous layer's output
        self.logit_previous = logit_previous
        return self.output_logits

    def update(self, targets, learning_rate=0.001):
        sigmoids = np.clip(sigmoid(self.output_logits), self.epsilon,
                           1 - self.epsilon)
        update_values = learning_rate * np.expand_dims(
            (sigmoids - targets), axis=1) * self.logit_previous
        self.weights[np.arange(self.num_neurons).reshape(-1, 1), self.
                     current_contexts, :] = np.clip(
                         self.weights[np.arange(self.num_neurons).
                                      reshape(-1, 1), self.current_contexts, :]
                         - np.transpose(update_values,
                                        (0, 2, 1)), -self.beta, self.beta)


class Model():
    def __init__(self, layers=[4, 4, 1], side_info_dim=784, epsilon=0.01):
        self.layers = []
        for idx, num_neurons in enumerate(layers):
            if idx == 0:
                # process base layer outputs
                layer = Layer(num_neurons=num_neurons,
                              input_dim=side_info_dim,
                              side_info_dim=side_info_dim)
            else:
                # process inner layer outputs
                layer = Layer(num_neurons=num_neurons,
                              input_dim=layers[idx - 1],
                              side_info_dim=side_info_dim)
            self.layers.append(layer)
        # squash inputs in base layer as suggested in paper
        self.base_layer = lambda x: (x * (1 - 2 * epsilon)) + epsilon
        self.learning_rate = 0.001

    def set_learning_rate(self, lr):
        self.learning_rate = lr

    def forward(self, inputs):
        out = self.base_layer(inputs)
        for l in self.layers:
            out = l.forward(out, inputs)
        return sigmoid(out)

    def update(self, targets):
        for l in self.layers:
            l.update(targets, self.learning_rate)


# %%
if __name__ == '__main__':
    m = Model(layers=[128, 128, 1])
    acc, conf_mat, prfs = get_mnist_metrics(m, batch_size=8, mnist_class=3)
