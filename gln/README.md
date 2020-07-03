# Gated Linear Networks
## Implementations in NumPy, PyTorch, TensorFlow and JAX

Python implementations of new family of neural networks from DeepMind's paper on [GLN](https://arxiv.org/pdf/1910.01526.pdf)

## TL;DR

We will look at the NumPy API. The other implementations have the same flow.

To define a GLN and have it trained on MNIST data.


```python
from gln_numpy import GLN
from test_mnist import get_mnist_metrics

model = GLN(layer_sizes=[4, 4, 1], input_size=784, context_size=784)
```

Assuming one batch of random inputs like MNIST data


```python
import numpy as np

train_X = np.random.normal(size=(784, 4))
context_inputs = train_X
train_Y = np.array([0, 1, 1, 0])
```

we can train the GLN (predict and update) in one step with


```python
output = model.predict(inputs=train_X, context_inputs=context_inputs, targets=train_Y)
```

To predict with the model, we just omit the `targets` parameter.


```python
pred = model.predict(inputs=train_X, context_inputs=context_inputs)
print(pred)
```

    [0.14558098 0.92031338 0.92967196 0.14376964]


To check that the model is learning, we can pass the same batch a few times to see the outputs get better.


```python
for i in range(5):
    output = model.predict(inputs=train_X, context_inputs=context_inputs, targets=train_Y)
    print('After iteration {}'.format(i+1), output)
```

    After iteration 1 [0.14558098 0.92031338 0.92967196 0.14376964]
    After iteration 2 [0.1276787  0.9298401  0.93720979 0.12630932]
    After iteration 3 [0.11318972 0.93737909 0.94335443 0.11212472]
    After iteration 4 [0.10120459 0.94352606 0.94847533 0.10035937]
    After iteration 5 [0.09113055 0.94864886 0.95281605 0.09044939]

