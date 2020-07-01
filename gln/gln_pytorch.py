# %%
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from test_mnist import get_mnist_metrics

if torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'


def data_transform(X, y):
    return torch.as_tensor(X, dtype=torch.float32).to(DEVICE), torch.as_tensor(
        y, dtype=torch.float32).to(DEVICE)


def result_transform(output):
    return output.detach().cpu().numpy()


class Neuron(nn.Module):
    def __init__(self,
                 input_dim=128,
                 side_info_dim=784,
                 context_dim=4,
                 mu=0.,
                 std=0.1):
        super(Neuron, self).__init__()
        self.v = nn.Parameter(torch.normal(mean=mu,
                                           std=std,
                                           size=(side_info_dim, context_dim)),
                              requires_grad=False)
        self.v /= torch.norm(self.v, dim=0, keepdim=True)
        self.b = nn.Parameter(torch.normal(mean=mu,
                                           std=std,
                                           size=(context_dim, 1)),
                              requires_grad=False)
        self.weights = nn.Parameter(
            torch.ones(size=(input_dim, 2**context_dim)) * (1 / input_dim),
            requires_grad=False)
        self.boolean_converter = nn.Parameter(torch.as_tensor(
            np.array([[2**i] for i in range(context_dim)])).float(),
                                              requires_grad=False)

    def forward(self, logit_previous, side_information):
        projection = torch.matmul(self.v.T, side_information)
        if projection.ndim == 1:
            projection = projection.reshape(-1, 1)
        self.current_contexts = torch.matmul(
            self.boolean_converter.T,
            (projection > self.b).float()).flatten().long()
        self.current_selected_weights = self.weights[:, self.current_contexts]
        self.output_logits = torch.mul(self.current_selected_weights,
                                       logit_previous).sum(axis=0)
        self.logit_previous = logit_previous
        return self.output_logits

    def update(self, targets, learning_rate=0.001):
        sigmoids = F.sigmoid(self.output_logits)
        update_value = learning_rate * (sigmoids -
                                        targets) * self.logit_previous
        for i in range(update_value.shape[-1]):
            self.weights[:, self.current_contexts[i]] -= update_value[:, i]

    def extra_repr(self):
        return f'input_dim={self.weights.size(0)}, context_dim={self.v.size(1)}'


class Layer(nn.Module):
    def __init__(self,
                 num_neurons=128,
                 input_dim=128,
                 side_info_dim=784,
                 epsilon=0.05,
                 beta=5):
        super(Layer, self).__init__()
        self.neurons = nn.ModuleList([
            Neuron(input_dim, side_info_dim)
            for i in range(max(1, num_neurons - 1))
        ])
        self.bias = nn.Parameter(torch.as_tensor(
            np.random.uniform(epsilon, 1 - epsilon)),
                                 requires_grad=False)

    def forward(self, logit_previous, side_information):
        output_logits = []
        if len(self.neurons) > 1:
            output_logits.append(
                torch.repeat_interleave(self.bias,
                                        logit_previous.size(-1)).to(DEVICE))

        for n in self.neurons:
            output_logits.append(n.forward(logit_previous, side_information))

        output = torch.stack(output_logits)
        return output

    def update(self, targets, learning_rate=0.01):
        for n in self.neurons:
            n.update(targets, learning_rate)

    def extra_repr(self):
        return f'bias={self.bias if len(self.neurons)>1 else None}'


class LayerVec(nn.Module):
    def __init__(self,
                 num_neurons=128,
                 input_dim=128,
                 side_info_dim=784,
                 context_dim=4,
                 mu=0.0,
                 std=0.1,
                 epsilon=0.05,
                 beta=1.5):
        super(LayerVec, self).__init__()

        self.num_neurons = num_neurons  # - 1
        self.bias = np.random.uniform(epsilon, 1 - epsilon)
        self.v = nn.Parameter(torch.as_tensor(
            np.random.normal(loc=mu,
                             scale=std,
                             size=(num_neurons, context_dim, side_info_dim))),
                              requires_grad=False)
        self.v /= torch.norm(self.v, dim=2, keepdim=True)
        self.b = nn.Parameter(torch.as_tensor(
            np.random.normal(loc=mu,
                             scale=std,
                             size=(num_neurons, context_dim, 1))),
                              requires_grad=False)
        self.boolean_converter = nn.Parameter(torch.as_tensor(
            np.array([[2**i] for i in range(context_dim)])),
                                              requires_grad=False)
        self.weights = nn.Parameter(
            torch.ones(size=(num_neurons, 2**context_dim, input_dim),
                       dtype=torch.float64) * (1 / input_dim),
            requires_grad=False)
        self.epsilon = epsilon
        self.beta = beta

    def forward(self, logit_previous, side_information):
        projection = torch.matmul(self.v, side_information)
        binary = (projection > self.b).int()
        self.current_contexts = torch.squeeze(
            torch.sum(binary * self.boolean_converter, dim=1))
        self.current_selected_weights = self.weights[torch.arange(
            self.num_neurons).reshape(-1, 1), self.current_contexts, :]
        self.output_logits = torch.matmul(self.current_selected_weights,
                                          logit_previous).diagonal(dim1=1,
                                                                   dim2=2)
        if self.num_neurons > 1:
            self.output_logits[0] = self.bias
        self.logit_previous = logit_previous
        return self.output_logits

    def update(self, targets, learning_rate=0.001):
        sigmoids = torch.clamp(F.sigmoid(self.output_logits), self.epsilon,
                               1 - self.epsilon)
        update_values = learning_rate * torch.unsqueeze(
            (sigmoids - targets), dim=1) * self.logit_previous
        self.weights[torch.arange(self.num_neurons).reshape(-1, 1), self.
                     current_contexts, :] = torch.clamp(
                         self.weights[torch.arange(self.num_neurons).
                                      reshape(-1, 1), self.current_contexts, :]
                         - update_values.permute(0, 2, 1), -self.beta,
                         self.beta)

    def extra_repr(self):
        return 'input_dim={}, neurons={}, context_dim={}, bias={}'.format(
            self.weights.size(2), self.v.size(0), self.v.size(1), self.bias)


class Model(nn.Module):
    def __init__(self,
                 layers=[4, 4, 1],
                 side_info_dim=784,
                 epsilon=0.01,
                 beta=5):
        super(Model, self).__init__()
        self.layers = []
        for idx, num_neurons in enumerate(layers):
            if idx == 0:
                # process base layer outputs
                layer = Layer(num_neurons=num_neurons,
                              input_dim=side_info_dim,
                              side_info_dim=side_info_dim,
                              epsilon=epsilon,
                              beta=beta)
            else:
                # process inner layer outputs
                layer = Layer(num_neurons=num_neurons,
                              input_dim=layers[idx - 1],
                              side_info_dim=side_info_dim,
                              epsilon=epsilon,
                              beta=beta)
            self.layers.append(layer)

        self.layers = nn.ModuleList(self.layers)
        # squash inputs in base layer
        self.base_layer = lambda x: (x * (1 - 2 * epsilon)) + epsilon
        self.learning_rate = 0.001

    def set_learning_rate(self, lr):
        self.learning_rate = lr

    def forward(self, inputs):
        out = self.base_layer(inputs)
        for l in self.layers:
            out = l.forward(out, inputs)
        return F.sigmoid(out)

    def update(self, targets):
        for l in self.layers:
            l.update(targets, self.learning_rate)


# %%
m = Model([128, 128, 128, 1]).to(DEVICE)
acc, conf, prfs = get_mnist_metrics(m,
                                    mnist_class=9,
                                    batch_size=16,
                                    data_transform=data_transform,
                                    result_transform=result_transform)
print(acc)

# %%
accuracies, conf_mats, prfs_mats = [], [], []
for i in range(10):
    m = Model([128, 128, 128, 128, 1]).to(DEVICE)
    acc, conf, prfs = get_mnist_metrics(m,
                                        mnist_class=i,
                                        batch_size=16,
                                        data_transform=data_transform,
                                        result_transform=result_transform)
    print(i, acc)
    accuracies.append(acc)
    conf_mats.append(conf)
    prfs_mats.append(prfs)

print(np.average(accuracies))
