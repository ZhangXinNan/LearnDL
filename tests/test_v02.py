"""LearnDL v0.2 regression checks (run from project root).

    python -m unittest discover -s tests -p 'test_v02.py' -v

Tests the public `dezero` package currently present in the repository.
"""
import gc
import unittest

import numpy as np

from dezero import Function, Variable, no_grad, using_config
from dezero.core_simple import Config


class Pair(Function):
    """A two-output operator: (x^2, 3*x)."""

    def forward(self, x):
        return x * x, 3 * x

    def backward(self, gy0, gy1):
        x = self.inputs[0].data
        return 2 * x * gy0 + 3 * gy1


class CoreV02Tests(unittest.TestCase):
    def test_leaf_backward(self):
        x = Variable(np.array(2.0))
        x.backward()
        np.testing.assert_allclose(x.grad, 1.0)

    def test_shared_variable(self):
        x = Variable(np.array(3.0))
        z = x * x + x
        z.backward()
        np.testing.assert_allclose(z.data, 12.0)
        np.testing.assert_allclose(x.grad, 7.0)

    def test_branch_generation(self):
        x = Variable(np.array(2.0))
        a = x * x
        z = a * a + a * a
        z.backward()
        np.testing.assert_allclose(x.grad, 64.0)

    def test_multi_output_both_used(self):
        x = Variable(np.array(2.0))
        a, b = Pair()(x)
        (a + b).backward()
        np.testing.assert_allclose(x.grad, 7.0)

    def test_multi_output_one_unused_but_live(self):
        x = Variable(np.array(2.0))
        a, b = Pair()(x)
        (2 * a).backward()
        np.testing.assert_allclose(x.grad, 8.0)
        self.assertIsNone(b.grad)

    def test_multi_output_one_reclaimed(self):
        x = Variable(np.array(2.0))
        a, b = Pair()(x)
        del b
        gc.collect()
        (2 * a).backward()
        np.testing.assert_allclose(x.grad, 8.0)

    def test_operator_forward_and_backward(self):
        cases = [
            (lambda x: x + 3, 5, 1), (lambda x: 3 + x, 5, 1),
            (lambda x: x - 3, -1, 1), (lambda x: 3 - x, 1, -1),
            (lambda x: x * 3, 6, 3), (lambda x: 3 * x, 6, 3),
            (lambda x: x / 4, .5, .25), (lambda x: 4 / x, 2, -1),
            (lambda x: x ** 3, 8, 12), (lambda x: -x, -2, -1),
        ]
        for operation, expected_y, expected_dx in cases:
            with self.subTest(operation=operation):
                x = Variable(np.array(2.0))
                y = operation(x)
                y.backward()
                np.testing.assert_allclose(y.data, expected_y)
                np.testing.assert_allclose(x.grad, expected_dx)

    def test_cleargrad(self):
        x = Variable(np.array(2.0))
        (x * x).backward()
        np.testing.assert_allclose(x.grad, 4)
        x.cleargrad()
        self.assertIsNone(x.grad)
        (x * x).backward()
        np.testing.assert_allclose(x.grad, 4)

    def test_no_grad_and_restore(self):
        x = Variable(np.array(2.0))
        with no_grad():
            y = x * x
            self.assertIsNone(y.creator)
        self.assertTrue(Config.enable_backprop)
        with self.assertRaisesRegex(RuntimeError, 'test exception'):
            with no_grad():
                raise RuntimeError('test exception')
        self.assertTrue(Config.enable_backprop)

    def test_binary_variable_gradients(self):
        # Two trainable operands: exercise BOTH gradients through /, * and -.
        x = Variable(np.array(2.0))
        y = Variable(np.array(3.0))
        z = x / y + x * y - y
        z.backward()
        np.testing.assert_allclose(z.data, 2/3 + 6 - 3)
        np.testing.assert_allclose(x.grad, 1/3 + 3)
        np.testing.assert_allclose(y.grad, -2/9 + 2 - 1)

    def test_retain_intermediate_grad(self):
        for retain in (False, True):
            with self.subTest(retain_grad=retain):
                x = Variable(np.array(2.0))
                a = x * x
                y = a * a
                y.backward(retain_grad=retain)
                np.testing.assert_allclose(x.grad, 32)
                if retain:
                    np.testing.assert_allclose(a.grad, 8)
                    np.testing.assert_allclose(y.grad, 1)
                else:
                    self.assertIsNone(a.grad)
                    self.assertIsNone(y.grad)

    def test_nested_no_grad_restores_prior_state(self):
        x = Variable(np.array(2.0))
        with no_grad():
            self.assertFalse(Config.enable_backprop)
            self.assertIsNone((x * x).creator)
            with using_config('enable_backprop', True):
                self.assertIsNotNone((x * x).creator)
            self.assertFalse(Config.enable_backprop)
        self.assertTrue(Config.enable_backprop)

    def test_matyas_gradient(self):
        x = Variable(np.array(1.0))
        y = Variable(np.array(2.0))
        z = .26 * (x ** 2 + y ** 2) - .48 * x * y
        z.backward()
        np.testing.assert_allclose(z.data, .34)
        np.testing.assert_allclose(x.grad, -.44)
        np.testing.assert_allclose(y.grad, .56)

    def test_goldstein_gradient_numerical(self):
        def goldstein(x, y):
            a = 1 + (x + y + 1) ** 2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)
            b = 30 + (2*x - 3*y) ** 2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2)
            return a * b
        x, y = Variable(np.array(.7)), Variable(np.array(-.3))
        z = goldstein(x, y)
        z.backward()
        eps = 1e-6
        gx = (goldstein(.7 + eps, -.3) - goldstein(.7 - eps, -.3)) / (2*eps)
        gy = (goldstein(.7, -.3 + eps) - goldstein(.7, -.3 - eps)) / (2*eps)
        np.testing.assert_allclose(x.grad, gx, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(y.grad, gy, rtol=1e-5, atol=1e-5)


if __name__ == '__main__':
    unittest.main()
