# Gated Linear Networks
## Implementations in NumPy, PyTorch, TensorFlow and JAX

Python implementations of new family of neural networks from DeepMind's paper on [GLN](https://arxiv.org/pdf/1910.01526.pdf)

To install, clone the repository and run

```pip install -e .```

We provide a generic wrapper for all four backends.

## Usage

A `GLN` model currently acts as a binary classifier by default. For `n` classes, our implementation creates n separate GLNs and trains together in a one-vs-all fashion to create an ensemble.

To play around with GLN models we have provided some utility function in `pygln.utils` that use the `MNIST` dataset.


```python
from pygln.utils import get_mnist
import numpy as np

X_train, y_train, X_test, y_test = get_mnist()
```

Let's first train a binary classifier. We will take class `3` as the target.


```python
y_train_3 = (y_train == 3)
y_test_3 = (y_test == 3)
```

To create a binary GLN classifier


```python
from pygln import GLN

model = GLN(backend='numpy', layer_sizes=[4, 4, 1], input_size=X_train.shape[1], learning_rate=1e-4)
```

To train the model with one pass of the data with a batch of 1 sample per iteration


```python
for idx in range(X_train.shape[0]):
    model.predict(input=X_train[idx], target=y_train_3[idx])
```

One can also use higher batch sizes for training.

To predict with the model, we just omit the `target` parameter.


```python
preds = []
for idx in range(X_test.shape[0]):
    preds.append(model.predict(X_test[idx]))

preds = np.vstack(preds)
```

Now we can check the accuracy of the trained model


```python
from sklearn.metrics import accuracy_score

accuracy_score(y_test_3, preds)
```




    0.9861



As we can see, the accuracy is quite high, given that we only made one pass through the data.

To train a model over the whole `MNIST` dataset, we create a `GLN` model with 10 classes. This creates 10 separate `GLN` binary classifiers, trains them together in a one-vs-all fashion and ensembles them for the output.


```python
mnist_model = GLN(backend='numpy', layer_sizes=[4, 4, 1], input_size=X_train.shape[1], num_classes=10, learning_rate=1e-4)

for idx in range(X_train.shape[0]):
    mnist_model.predict(input=X_train[idx], target=[y_train[idx]])

preds = []
for idx in range(X_test.shape[0]):
    preds.append(mnist_model.predict(X_test[idx]))
accuracy_score(y_test, np.vstack(preds))
```




    0.9409



We have provided `utils.evaluate` to do these tests for the `MNIST` dataset. To train a GLN as a binary classifier for a particular class


```python
from pygln.utils import evaluate_mnist

mnist_model = GLN(backend='numpy', layer_sizes=[4, 4, 1], input_size=784, learning_rate=1e-4)
evaluate_mnist(mnist_model, mnist_class=3, batch_size=4)
```

    100%|██████████| 15000/15000 [00:10<00:00, 1366.94it/s]
    100%|██████████| 2500/2500 [00:01<00:00, 2195.59it/s]





    98.69



and to train on all classes


```python
from pygln.utils import evaluate_mnist

mnist_model = GLN(backend='numpy', layer_sizes=[4, 4, 1], input_size=784, num_classes=10, learning_rate=1e-4)
evaluate_mnist(mnist_model, batch_size=4)
```

    100%|██████████| 15000/15000 [00:35<00:00, 418.21it/s]
    100%|██████████| 2500/2500 [00:03<00:00, 764.10it/s]





    94.69


