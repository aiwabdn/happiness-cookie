# Gated Linear Networks

Let's start with the concept of a neuron.

Each neuron has the following features:
* A set of **weight** vectors. Each of these vectors belongs to one out of a set of contexts that the neuron maintains. These vectors are grouped in an array and accessed by their index.
* A **context function** that maps provided side information (in our case the inputs themselves) to an index of the weight matrix. This context function is realised by two parts:
    1. A matrix of vectors **v**, randomly sampled from $$N(0, 0.1)$$, that project each input vector onto the number line.
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

context_dim = 4
side_info_dim = 784
input_dim = 128
```

Let's also define some random side information and input vectors.


```python
side_info = np.random.normal(size=(784, 1))
previous_layer_output = np.random.normal(size=(128, 1))
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
     [1]]


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

    9


So for our given side information, we are going to use the weight vector at index `current_context`.

Now let's define the weight vectors that we are going to choose from.
Given that our context dimension is `4` and that we derive a 4-bit binary index for each context, we can have `2^4 = 16` different weight vectors to choose from. The neuron will take the selected weight vector and multiply it with the output from the previous layer to produce the output logit. We can straightaway put all the vectors into one matrix and define it as


```python
weights = np.ones(shape=(2 ** context_dim, input_dim)) * (1 / input_dim)
```

where we have used uniform initialisation of the weights as suggested in the paper.

Having determined the specific context to use, we can choose the weight vector and process the input as


```python
output_logit = weights[current_context].dot(previous_layer_output)
print(output_logit)
```

    [-0.03063838]

