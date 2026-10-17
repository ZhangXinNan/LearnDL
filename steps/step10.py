
import numpy as np
import unittest

from step09 import Variable, square, exp
from step04 import numerical_diff


class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data, expected)

    def test_backward(self):
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        expected = np.array(6.0)
        self.assertEqual(x.grad, expected)

    def test_gradient_check(self):
        # x = Variable(np.random.rand(1))
        x = Variable(np.array(2.0))
        y = square(x)
        y.backward()
        num_grad = numerical_diff(square, x)
        flg = np.allclose(x.grad, num_grad)
        self.assertTrue(flg)

    def test_composite_backward(self):
        x = Variable(np.array(0.5))
        y = square(exp(square(x)))
        y.backward()

        expected = 4 * x.data * np.exp(2 * x.data ** 2)
        np.testing.assert_allclose(x.grad, expected)

    def test_composite_gradient_check(self):
        x = Variable(np.array(0.5))
        f = lambda v: square(exp(square(v)))

        y = f(x)
        y.backward()

        num_grad = numerical_diff(f, x)
        np.testing.assert_allclose(
            x.grad, num_grad, rtol=1e-5, atol=1e-6
        )


if __name__ == '__main__':
    # python step10.py
    unittest.main()

