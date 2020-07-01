# Gated Linear Networks

Let's start with the concept of a neuron.

Each neuron has the following features:
* A set of **weight** vectors. Each of these vectors belongs to one out of a set of contexts that the neuron maintains. These vectors are grouped in an array and accessed by their index.
* A **context function** that maps provided side information (in our case the inputs themselves) to an index of the weight matrix. This context function is realised by two parts:
    1. A matrix of vectors **v**, randomly sampled from $N(0, 0.1)$, that project each input vector onto the number line.
    2. A vector of values **b**, again randomly sampled from a Gaussian, that are used to decide which side of the halfspace the given input falls on.

Given these features, a neuron
1. projects the side information onto the number line for different halfspace hypotheses.
2. produces a binary vector from the numbers by checking whether they are greater than a fixed set of values to decide which side of the halfspace the input falls on.
3. converts the binary vector to an index to choose which context to use for this particular input.
4. multiplies the chosen weight vector with the output from the previous layer to produce the output.
5. computes the loss directly by comparing its output to the ground truth and updates the chosen weight vector.

Let's implement each of these parts one by one.
For this hypothetical neuron, we are going to assume `context_dim = 4` for each neuron, side information dimension to be `side_info_dim = 784` (the length of each data point in the MNIST dataset) and the input dimension to be `input_dim = 128`.


```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

learning_rate = 0.01
    
context_dim = 4
side_info_dim = 784
input_dim = 128
```

Let's also define some random side information, input and target vectors.


```python
side_info = np.random.normal(size=(784, 1))
previous_layer_output = np.random.normal(size=(128, 1))
target = np.array([0])
```

Now let's define the parts of the context function. The projection matrix will have dimensions `(context_dim, side_info_dim)`, the threshold vector will have dimensions `(context_dim, 1)`. Additionally, as mentioned in the paper, the projection vectors will be scaled by their L2 norms.


```python
v = np.random.normal(loc=0, scale=0.1, size=(context_dim, side_info_dim))
v /= np.linalg.norm(v, ord=2, axis=1, keepdims=True)
b = np.random.normal(loc=0, scale=0.1, size=(context_dim, 1))
```

Given a side information, we can compute the binary vector that will serve as the index of the context based weight vector as


```python
binary_context = (v.dot(side_info) > b).astype(np.int)
print(binary_context)
```

    [[1]
     [0]
     [0]
     [0]]


To convert this boolean into an integer index, and in keeping with vectorised implementations, our neurons will also have a converter matrix that holds powers of `2` as such


```python
boolean_converter = np.array([[2 ** i] for i in range(context_dim)])
print(boolean_converter)
```

    [[1]
     [2]
     [4]
     [8]]


This will help us a lot when it comes to batched vectorised implementations.
We can use these bits to find our context specific weight vector index as


```python
current_context = np.squeeze(np.sum(binary_context * boolean_converter, axis=0))
print(current_context)
```

    1


So for our given side information, we are going to use the weight vector at index `current_context`.

Now let's define the weight vectors that we are going to choose from.
Given that our context dimension is `4` and that we derive a 4-bit binary index for each context, we can have `2^4 = 16` different weight vectors to choose from. The neuron will take the selected weight vector and multiply it with the output from the previous layer to produce the output logit. We can straightaway put all the vectors into one matrix and define it as


```python
weights = np.ones(shape=(2 ** context_dim, input_dim)) * (1 / input_dim)
```

where we have used uniform initialisation of the weights as suggested in the paper.

Having determined the specific context to use, we can choose the weight vector and produce the output as


```python
output_logit = weights[current_context].dot(previous_layer_output)
output = sigmoid(output_logit)
print(output)
```

    [0.48991319]


Having computed the output, we now need to compute the error between the neuron's prediction and the ground truth and update the weights that were used for this particular input. As suggested in the paper, for optimal learning we will be clipping the outputs used for updates and the weights themselves within certain ranges.


```python
update = np.clip((output - target) * previous_layer_output, 0.01, 0.99)
weights[current_context] -= learning_rate * update.flatten()
weights = np.clip(weights, -5, 5)
```

And there we have the full action cycle of one neuron. Let's put all of this together in a class in a way that can also handle batches of inputs and side_information.


```python
class Neuron():
    def __init__(self,
                 input_dim=128,
                 side_info_dim=784,
                 context_dim=4,
                 mu=0.0,
                 std=0.1,
                 epsilon=0.01,
                 beta=5):
        # context function for halfspace gating
        self.v = np.random.normal(loc=mu,
                                  scale=std,
                                  size=(context_dim, side_info_dim))
        # scale by norm
        self.v = self.v / np.linalg.norm(self.v, ord=2, axis=1, keepdims=True)
        # constant values for halfspace gating
        self.b = np.random.normal(size=(context_dim, 1))
        # weights for the neuron
        self.weights = np.ones(shape=(2**context_dim,
                                      input_dim)) * (1 / input_dim)
        # array to convert binary context to index
        self.boolean_converter = np.array([[2**i] for i in range(context_dim)])
        # clip values
        self.epsilon = epsilon
        self.beta = beta

    def forward(self, logit_previous, side_information):
        # project side information and determine context index
        projection = self.v.dot(side_information)
        if projection.ndim == 1:
            projection = projection.reshape(-1, 1)
        binary = (projection > self.b).astype(np.int)
        self.current_contexts = np.squeeze(
            np.sum(binary * self.boolean_converter, axis=0))

        # select weights for current batch
        self.current_selected_weights = self.weights[self.current_contexts, :]
        # compute logit output
        self.output_logits = self.current_selected_weights.dot(
            logit_previous).diagonal()
        self.logit_previous = logit_previous
        return self.output_logits

    def update(self, targets, learning_rate=0.001):
        # compute output and clip
        sigmoids = np.clip(sigmoid(self.output_logits), self.epsilon,
                           1 - self.epsilon)
        # compute update
        update_value = learning_rate * (sigmoids -
                                        targets) * self.logit_previous
        # iterate through selected contexts and update
        for i in range(update_value.shape[-1]):
            self.weights[self.current_contexts[i], :] -= update_value[:, i]
        # clip weights
        self.weights = np.clip(self.weights, -self.beta, self.beta)
```
