# Gated Linear Networks
## Implementations in NumPy, PyTorch, TensorFlow and JAX

Python implementations of new family of neural networks from DeepMind's paper on [GLN](https://arxiv.org/pdf/1910.01526.pdf)

## TL;DR

We will look at the NumPy API. The other implementations have the same flow.

To define a GLN and have it trained on MNIST data.


```python
from gln_numpy import GLN

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

    [0.13668085 0.92979607 0.91990108 0.10906544]


To check that the model is learning, we can pass the same batch a few times to see the outputs get better.


```python
for i in range(5):
    output = model.predict(inputs=train_X, context_inputs=context_inputs, targets=train_Y)
    print('After iteration {}'.format(i+1), output)
```

    After iteration 1 [0.13668085 0.92979607 0.91990108 0.10906544]
    After iteration 2 [0.1193458  0.93671518 0.92867751 0.09754806]
    After iteration 3 [0.10552609 0.94249631 0.93583505 0.08790868]
    After iteration 4 [0.09421555 0.94740545 0.94179999 0.07972696]
    After iteration 5 [0.08478159 0.95162827 0.94685394 0.07270384]


Some helper functions are provided in `test_mnist`. We can train a model on a particular MNIST class with one pass of the data. 


```python
from test_mnist import get_mnist_metrics

model = GLN(layer_sizes=[4, 4, 1], input_size=784, context_size=784)
acc, conf_mat, prfs = get_mnist_metrics(model, batch_size=8, mnist_class=3)
print()
print('Accuracy:', acc)
print('Confusion matrix:\n', conf_mat)
print('Prec-Rec-F:\n', prfs)
```

    100%|██████████| 7500/7500 [00:04<00:00, 1658.96it/s]
    100%|██████████| 1250/1250 [00:00<00:00, 4915.80it/s]
    
    Accuracy: 97.81
    Confusion matrix:
            not_3  is_3
    not_3   8945    45
    is_3     174   836
    Prec-Rec-F:
                      not_3         is_3
    precision     0.980919     0.948922
    recall        0.994994     0.827723
    fscore        0.987907     0.884188
    support    8990.000000  1010.000000

