"""v0.5 (Steps 42-46) checks: Layer, Linear, MLP, and SGD.

Run from repository root: python -m unittest discover -s tests -p 'test_v05.py' -v
"""
import unittest
import numpy as np

from dezero import Variable
from dezero.layers import Linear
from dezero.models import MLP
from dezero.optimizers import SGD
import dezero.functions as F


class V05Tests(unittest.TestCase):
    def test_linear_lazy_parameters_and_mlp(self):
        np.random.seed(11)
        model = MLP((4, 3))
        self.assertIsNone(model.l0.W.data)
        x = np.ones((5, 2), dtype=np.float32)
        self.assertEqual(model(x).shape, (5, 3))
        self.assertEqual(model.l0.W.data.shape, (2, 4))
        self.assertEqual(model.l1.W.data.shape, (4, 3))
        self.assertEqual(len(list(model.params())), 4)

    def test_sgd_update_matches_gradient_descent(self):
        layer = Linear(1, in_size=1)
        layer.W.data[:] = 2.
        layer.b.data[:] = 1.
        x = np.array([[1.]], dtype=np.float32)
        target = np.array([[0.]], dtype=np.float32)
        loss = F.mean_squared_error(layer(x), target)
        layer.cleargrads()
        loss.backward()
        old_weights = [p.data.copy() for p in layer.params()]
        grads = [p.grad.data.copy() for p in layer.params()]
        SGD(0.1).setup(layer).update()
        for p, old, grad in zip(layer.params(), old_weights, grads):
            np.testing.assert_allclose(p.data, old - .1 * grad, rtol=1e-6)

    def test_sgd_learns_simple_linear_regression(self):
        np.random.seed(5)
        x = np.array([[-1.], [0.], [1.], [2.]], dtype=np.float32)
        target = 3*x + 1
        layer = Linear(1, in_size=1)
        opt = SGD(0.05).setup(layer)
        losses = []
        for _ in range(100):
            loss = F.mean_squared_error(layer(x), target)
            losses.append(float(loss.data))
            layer.cleargrads()
            loss.backward()
            opt.update()
        self.assertLess(losses[-1], losses[0] * 1e-3)
        np.testing.assert_allclose(layer.W.data, [[3]], atol=0.05)
        np.testing.assert_allclose(layer.b.data, [1], atol=0.05)


if __name__ == '__main__':
    unittest.main()
