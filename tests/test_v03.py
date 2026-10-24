"""LearnDL v0.3 高阶自动微分测试。

放到 tests/test_v03.py 后，在项目根目录运行：
    python -m unittest discover -s tests -p 'test_v03.py' -v

这组测试是 v0.3 的严格验收用例：未修复时会报告真实失败，
修复 core.py 中 Square/Div/Pow.backward 后应全部通过。
"""
import unittest

import numpy as np

from dezero.core import Config, Variable, no_grad, setup_variable, square
from dezero import functions as F
from dezero.utils import get_dot_graph

# 单独测试 v0.3 的 core，不受 dezero/__init__.py 中版本切换影响。
setup_variable()


class V03AutogradTests(unittest.TestCase):
    @staticmethod
    def grad2(function, x_value):
        x = Variable(np.array(float(x_value)))
        y = function(x)
        y.backward(create_graph=True)
        gx = x.grad
        first = gx.data.copy()
        x.cleargrad()
        gx.backward()
        return first, None if x.grad is None else x.grad.data.copy()

    def test_backward_gradient_is_variable(self):
        x = Variable(np.array(2.0))
        (x * x).backward()
        self.assertIsInstance(x.grad, Variable)
        np.testing.assert_allclose(x.grad.data, 4.0)

    def test_default_backward_does_not_build_gradient_graph(self):
        x = Variable(np.array(2.0))
        (x * x).backward(create_graph=False)
        self.assertIsNone(x.grad.creator)
        self.assertTrue(Config.enable_backprop)

    def test_create_graph_retains_higher_order_dependency(self):
        x = Variable(np.array(2.0))
        (x * x).backward(create_graph=True)
        self.assertIsNotNone(x.grad.creator)
        self.assertTrue(Config.enable_backprop)

    def test_power_second_derivative(self):
        first, second = self.grad2(lambda x: x**4, 2.0)
        np.testing.assert_allclose(first, 32.0)
        self.assertIsNotNone(second, 'x**4 的一阶梯度未连接到 x')
        np.testing.assert_allclose(second, 48.0)

    def test_square_second_derivative(self):
        first, second = self.grad2(square, 2.0)
        np.testing.assert_allclose(first, 4.0)
        self.assertIsNotNone(second)
        np.testing.assert_allclose(second, 2.0)

    def test_division_second_derivative(self):
        first, second = self.grad2(lambda x: 1 / x, 2.0)
        np.testing.assert_allclose(first, -0.25)
        self.assertIsNotNone(second)
        np.testing.assert_allclose(second, 0.25)

    def test_multiplication_second_derivative(self):
        first, second = self.grad2(lambda x: x*x*x*x, 2.0)
        np.testing.assert_allclose(first, 32.0)
        np.testing.assert_allclose(second, 48.0)

    def test_sin_second_derivative(self):
        first, second = self.grad2(F.sin, 1.0)
        np.testing.assert_allclose(first, np.cos(1.0))
        np.testing.assert_allclose(second, -np.sin(1.0))

    def test_cos_second_derivative(self):
        first, second = self.grad2(F.cos, 1.0)
        np.testing.assert_allclose(first, -np.sin(1.0))
        np.testing.assert_allclose(second, -np.cos(1.0))

    def test_tanh_second_derivative(self):
        t = np.tanh(1.0)
        first, second = self.grad2(F.tanh, 1.0)
        np.testing.assert_allclose(first, 1-t*t)
        np.testing.assert_allclose(second, -2*t*(1-t*t))

    def test_composed_second_derivative(self):
        v = 0.7
        s, c, t = np.sin(v), np.cos(v), np.tanh(v)
        expected = 12*v*v - s*t + 2*c*(1-t*t) - 2*s*t*(1-t*t)
        first, second = self.grad2(lambda x: x**4 + F.sin(x)*F.tanh(x), v)
        self.assertIsNotNone(second)
        np.testing.assert_allclose(second, expected, rtol=1e-10)

    def test_double_backprop_step36(self):
        x = Variable(np.array(2.0))
        y = x**2
        y.backward(create_graph=True)
        gx = x.grad
        x.cleargrad()
        z = gx**3 + y
        z.backward()
        np.testing.assert_allclose(x.grad.data, 100.0)

    def test_nograd_restores_setting(self):
        x = Variable(np.array(2.0))
        with no_grad():
            y = x**4
            self.assertIsNone(y.creator)
        self.assertTrue(Config.enable_backprop)

    def test_dot_graph_contains_operations(self):
        x = Variable(np.array(2.0), name='x')
        y = F.sin(x*x)
        graph = get_dot_graph(y)
        self.assertIn('digraph g', graph)
        self.assertIn('Mul', graph)
        self.assertIn('Sin', graph)


if __name__ == '__main__':
    unittest.main()
