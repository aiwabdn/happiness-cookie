import numpy as np
import unittest

from pygln import GLN, utils


class TestReadme(unittest.TestCase):

    def _test_backend(self, backend):
        X_train, y_train, X_test, y_test = utils.get_mnist()
        model = GLN(
            backend=backend, layer_sizes=[4, 4, 1], input_size=X_train.shape[1], num_classes=10
        )

        output = model.predict(X_train[:1])
        self.assertEqual(output.dtype, y_test.dtype)
        self.assertEqual(output.shape, (1,))

        output = model.predict(X_train[:10], target=y_train[:10])
        self.assertEqual(output.dtype, y_train.dtype)
        self.assertEqual(output.shape, (10,))

        output = model.predict(X_train[:4], target=y_train[:4], return_probs=True)
        self.assertTrue(np.issubdtype(output.dtype, np.floating))
        self.assertEqual(output.shape, (4, 10))

    def test_jax(self):
        self._test_backend(backend='jax')

    def test_numpy(self):
        self._test_backend(backend='numpy')

    def test_pytorch(self):
        self._test_backend(backend='pytorch')

    def test_tf(self):
        self._test_backend(backend='tf')
