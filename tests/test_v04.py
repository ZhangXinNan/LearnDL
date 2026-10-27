"""LearnDL v0.4 regression tests (textbook Steps 37–41).

Install only NumPy. From the repository root:
    python -m unittest discover -s tests -p 'test_v04.py' -v

The tests intentionally target the full `dezero.core` implementation rather
than `core_simple`, which is preserved for the earlier book chapters.
"""

import unittest

import numpy as np

from dezero.core import Variable, setup_variable
import dezero.functions as F

setup_variable()


class TestV04Tensors(unittest.TestCase):
    """Check shape transformations and gradients using central differences."""

    @staticmethod
    def _numerical_grad(fn, arrays, index, eps=1e-5):
        raw = [np.asarray(a, dtype=np.float64).copy() for a in arrays]
        grad = np.zeros_like(raw[index])
        for position in np.ndindex(raw[index].shape):
            raw[index][position] += eps
            y1 = fn(*[Variable(a.copy()) for a in raw]).data.sum()
            raw[index][position] -= 2 * eps
            y0 = fn(*[Variable(a.copy()) for a in raw]).data.sum()
            raw[index][position] += eps
            grad[position] = (y1 - y0) / (2 * eps)
        return grad

    def assert_gradients(self, fn, *arrays):
        values = [np.asarray(a, dtype=np.float64) for a in arrays]
        variables = [Variable(a.copy()) for a in values]
        y = fn(*variables)
        F.sum(y).backward()
        for i, var in enumerate(variables):
            with self.subTest(input_index=i):
                self.assertIsNotNone(var.grad)
                self.assertEqual(var.grad.shape, values[i].shape)
                expected = self._numerical_grad(fn, values, i)
                np.testing.assert_allclose(var.grad.data, expected,
                                           rtol=2e-5, atol=2e-6)

    def test_reshape(self):
        x = np.arange(6.).reshape(2, 3)
        c = np.arange(6.).reshape(3, 2) + 1
        self.assert_gradients(lambda a: F.reshape(a, (3, 2)) * c, x)
        self.assertEqual(Variable(x).reshape(3, 2).shape, (3, 2))

    def test_transpose(self):
        x = np.arange(24.).reshape(2, 3, 4) / 10
        c = np.arange(24.).reshape(4, 2, 3) / 7
        self.assert_gradients(lambda a: F.transpose(a, (2, 0, 1)) * c, x)
        self.assert_gradients(lambda a: F.transpose(a, (-1, 0, 1)) * c, x)
        self.assertEqual(Variable(np.ones((2, 3))).T.shape, (3, 2))

    def test_sum_axes_and_keepdims(self):
        x = np.arange(24.).reshape(2, 3, 4) / 10
        for axis, keepdims in [(None, False), (0, False), (-1, False),
                               ((0, 2), False), ((0, 2), True)]:
            with self.subTest(axis=axis, keepdims=keepdims):
                y = F.sum(Variable(x), axis=axis, keepdims=keepdims)
                np.testing.assert_allclose(y.data, x.sum(axis=axis, keepdims=keepdims))
                self.assert_gradients(lambda a: F.sum(a, axis, keepdims), x)

    def test_sum_to(self):
        x = np.arange(24.).reshape(2, 3, 4) / 10
        self.assert_gradients(lambda a: F.sum_to(a, (3, 1)), x)
        y = F.sum_to(Variable(x), (3, 1))
        np.testing.assert_allclose(y.data, x.sum(axis=(0, 2)).reshape(3, 1))

    def test_broadcast_to(self):
        x = np.arange(3.).reshape(3, 1)
        self.assert_gradients(lambda a: F.broadcast_to(a, (2, 3, 4)), x)
        self.assertEqual(F.broadcast_to(Variable(x), (2, 3, 4)).shape, (2, 3, 4))

    def test_broadcast_binary_operators(self):
        x = np.arange(24.).reshape(2, 3, 4) / 25 + 0.5
        b = (np.arange(3.).reshape(3, 1) + 3) / 2
        cases = [('add', lambda a, v: a + v),
                 ('sub', lambda a, v: a - v),
                 ('mul', lambda a, v: a * v),
                 ('div', lambda a, v: a / v)]
        for name, op in cases:
            with self.subTest(operator=name):
                self.assert_gradients(op, x, b)

    def test_matmul_2d(self):
        x = np.arange(6.).reshape(2, 3) / 10
        W = np.arange(12.).reshape(3, 4) / 9
        coeff = np.arange(8.).reshape(2, 4) / 8 + .5
        self.assert_gradients(lambda a, b: F.matmul(a, b) * coeff, x, W)

    def test_linear_bias(self):
        x = np.arange(6.).reshape(2, 3) / 10
        W = np.arange(12.).reshape(3, 4) / 9
        b = np.array([.1, .2, .3, .4])
        self.assert_gradients(lambda a, w, bias: F.linear(a, w, bias), x, W, b)

    def test_second_derivative_through_broadcast(self):
        x = Variable(np.array([[1., 2., 3.], [4., 5., 6.]]))
        b = Variable(np.array([2., 3., 4.]))
        y = F.sum((x + b) ** 2)
        y.backward(create_graph=True)
        first = b.grad
        np.testing.assert_allclose(first.data, np.array([18., 26., 34.]))
        b.cleargrad()
        F.sum(first).backward()
        np.testing.assert_allclose(b.grad.data, [4., 4., 4.])


if __name__ == '__main__':
    unittest.main()
