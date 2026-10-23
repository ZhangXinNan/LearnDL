"""LearnDL core_simple v0.2 tests.

运行方式（在 LearnDL 项目根目录）：
    python -m unittest discover -s tests -p 'test_v02.py' -v

本文件只测试 dezero.core_simple，避免 __init__.py 在 core/core_simple 间
切换时混用两套 Variable / Function。

测试原则：
1. 当前已经实现的功能严格断言，失败就说明需要排查。
2. 书上尚未实现的广播反传、多输出缺失梯度统一标记 expectedFailure。
3. 不为了让测试通过而修改教学用的 core_simple.py。
"""

import gc
import unittest
import weakref

import numpy as np

import dezero
from dezero.core_simple import (
    Config,
    Function,
    Square,
    Variable,
    add,
    as_array,
    as_variable,
    div,
    mul,
    neg,
    no_grad,
    pow as dz_pow,
    rdiv,
    rsub,
    setup_variable,
    square,
    sub,
    using_config,
)

# 即使 dezero/__init__.py 选择了 core，也可以单独测试 core_simple。
setup_variable()


class Pair(Function):
    """用来研究多输出计算图：f(x) = (x², 3x)。"""

    def forward(self, x):
        return x * x, 3 * x

    def backward(self, gy0, gy1):
        x = self.inputs[0].data
        return 2 * x * gy0 + 3 * gy1


class VariableAndUtilityTests(unittest.TestCase):
    """变量数据、属性、转换函数与基本协议。"""

    def test_package_exports_simple_core_when_selected(self):
        if dezero.is_simple_core:
            self.assertIs(dezero.Variable, Variable)
            self.assertIs(dezero.Function, Function)

    def test_as_array_converts_scalar_and_preserves_ndarray(self):
        for value in (2, 2.0, np.float32(1.5)):
            with self.subTest(value=value):
                self.assertIsInstance(as_array(value), np.ndarray)
        array = np.array([1.0, 2.0])
        self.assertIs(as_array(array), array)
        self.assertIsNone(as_array(None))

    def test_as_variable_wraps_data_and_is_idempotent(self):
        x = as_variable(2.0)
        self.assertIsInstance(x, Variable)
        np.testing.assert_allclose(x.data, 2.0)
        self.assertIs(as_variable(x), x)
        values = np.array([1.0, 2.0])
        self.assertIs(as_variable(values).data, values)

    def test_variable_requires_ndarray_or_none(self):
        for bad in (2, 2.0, [1, 2], (1, 2)):
            with self.subTest(value=bad), self.assertRaises(TypeError):
                Variable(bad)
        self.assertIsNone(Variable(None).data)

    def test_variable_properties_and_len(self):
        x = Variable(np.ones((2, 3), dtype=np.float32), name='x')
        self.assertEqual(x.name, 'x')
        self.assertEqual(x.shape, (2, 3))
        self.assertEqual(x.ndim, 2)
        self.assertEqual(x.size, 6)
        self.assertEqual(x.dtype, np.dtype('float32'))
        self.assertEqual(len(x), 2)
        self.assertIsNone(x.grad)
        self.assertIsNone(x.creator)
        self.assertEqual(x.generation, 0)
        self.assertGreater(Variable.__array_priority__, 0)

    def test_variable_repr(self):
        self.assertEqual(repr(Variable(None)), 'variable(None)')
        self.assertEqual(repr(Variable(np.array(2.0))), 'variable(2.0)')
        self.assertIn('\n', repr(Variable(np.ones((2, 2)))))

    def test_setup_variable_is_idempotent(self):
        original_add = Variable.__add__
        setup_variable()
        self.assertIs(Variable.__add__, original_add)

    def test_function_abstract_forward_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            Function()(Variable(np.array(2.0)))


class OperatorTests(unittest.TestCase):
    """每个一元/二元算子的前向值和反向梯度。"""

    def test_operator_forward_and_backward_with_python_scalar(self):
        cases = [
            ('add', lambda x: x + 3, 5, 1),
            ('radd', lambda x: 3 + x, 5, 1),
            ('sub', lambda x: x - 3, -1, 1),
            ('rsub', lambda x: 3 - x, 1, -1),
            ('mul', lambda x: x * 3, 6, 3),
            ('rmul', lambda x: 3 * x, 6, 3),
            ('div', lambda x: x / 4, .5, .25),
            ('rdiv', lambda x: 4 / x, 2, -1),
            ('pow', lambda x: x ** 3, 8, 12),
            ('neg', lambda x: -x, -2, -1),
        ]
        for name, operation, expected_y, expected_dx in cases:
            with self.subTest(operation=name):
                x = Variable(np.array(2.0))
                y = operation(x)
                y.backward()
                np.testing.assert_allclose(y.data, expected_y)
                np.testing.assert_allclose(x.grad, expected_dx)

    def test_two_variable_gradients_for_all_binary_operators(self):
        cases = [
            ('add', lambda x, y: x + y, 5., 1., 1.),
            ('sub', lambda x, y: x - y, -1., 1., -1.),
            ('mul', lambda x, y: x * y, 6., 3., 2.),
            ('div', lambda x, y: x / y, 2./3., 1./3., -2./9.),
        ]
        for name, operation, expected_z, expected_dx, expected_dy in cases:
            with self.subTest(operation=name):
                x = Variable(np.array(2.0))
                y = Variable(np.array(3.0))
                z = operation(x, y)
                z.backward()
                np.testing.assert_allclose(z.data, expected_z)
                np.testing.assert_allclose(x.grad, expected_dx)
                np.testing.assert_allclose(y.grad, expected_dy)

    def test_function_wrappers(self):
        cases = [
            ('square', lambda x: square(x), 4., 4.),
            ('add', lambda x: add(x, 3), 5., 1.),
            ('sub', lambda x: sub(x, 3), -1., 1.),
            ('rsub', lambda x: rsub(x, 3), 1., -1.),
            ('mul', lambda x: mul(x, 3), 6., 3.),
            ('div', lambda x: div(x, 4), .5, .25),
            ('rdiv', lambda x: rdiv(x, 4), 2., -1.),
            ('pow', lambda x: dz_pow(x, 3), 8., 12.),
            ('neg', lambda x: neg(x), -2., -1.),
        ]
        for name, operation, expected_y, expected_dx in cases:
            with self.subTest(wrapper=name):
                x = Variable(np.array(2.0))
                y = operation(x)
                y.backward()
                np.testing.assert_allclose(y.data, expected_y)
                np.testing.assert_allclose(x.grad, expected_dx)

    def test_pow_exponents(self):
        for exponent, value, derivative in (
            (0, 1., 0.), (1, 2., 1.), (2, 4., 4.),
            (3, 8., 12.), (-1, .5, -.25), (.5, np.sqrt(2), .5 / np.sqrt(2)),
        ):
            with self.subTest(exponent=exponent):
                x = Variable(np.array(2.0))
                y = x ** exponent
                y.backward()
                np.testing.assert_allclose(y.data, value)
                np.testing.assert_allclose(x.grad, derivative)

    def test_numpy_scalar_on_left(self):
        x = Variable(np.array(2.0))
        y = np.array(3.0) + x
        self.assertIsInstance(y, Variable)
        np.testing.assert_allclose(y.data, 5.0)
        y.backward()
        np.testing.assert_allclose(x.grad, 1.0)

    def test_array_elementwise_gradients_without_broadcast(self):
        x_data = np.array([1., 2., -3.])
        w_data = np.array([4., 5., 6.])
        x = Variable(x_data.copy())
        w = Variable(w_data.copy())
        y = (x * w + x ** 2) / w
        np.testing.assert_allclose(y.data, x_data + x_data**2 / w_data)
        y.backward()  # 元素级输出的默认上游梯度为全 1
        np.testing.assert_allclose(x.grad, 1 + 2 * x_data / w_data)
        np.testing.assert_allclose(w.grad, -(x_data**2) / (w_data**2))
        self.assertEqual(x.grad.shape, x.data.shape)
        self.assertEqual(w.grad.shape, w.data.shape)

    def test_broadcast_forward_only(self):
        # NumPy 广播的前向计算有效；这里不要求尚未实现的反传求和。
        x = Variable(np.ones((2, 3)))
        bias = Variable(np.array([1., 2., 3.]))
        y = x + bias
        np.testing.assert_allclose(y.data, [[2., 3., 4.], [2., 3., 4.]])


class AutogradGraphTests(unittest.TestCase):
    """动态计算图、反向遍历、梯度累积、弱引用与配置。"""

    def test_leaf_backward_initializes_ones(self):
        for data in (np.array(2.0), np.array([1., 2., 3.])):
            with self.subTest(shape=data.shape):
                x = Variable(data)
                x.backward()
                np.testing.assert_array_equal(x.grad, np.ones_like(data))

    def test_graph_creator_and_generation(self):
        x = Variable(np.array(2.0))
        y = square(x)
        z = y + x
        self.assertIsNone(x.creator)
        self.assertEqual(x.generation, 0)
        self.assertIsNotNone(y.creator)
        self.assertIsNotNone(z.creator)
        self.assertEqual(y.creator.generation, 0)
        self.assertEqual(y.generation, 1)
        self.assertEqual(z.creator.generation, 1)
        self.assertEqual(z.generation, 2)
        self.assertIs(y.creator.inputs[0], x)

    def test_chain_rule(self):
        x = Variable(np.array(2.0))
        z = 3 * x ** 2 + 2 * x - 1
        z.backward()
        np.testing.assert_allclose(z.data, 15.)
        np.testing.assert_allclose(x.grad, 14.)

    def test_shared_variable_gradients_are_added(self):
        x = Variable(np.array(3.0))
        z = x * x + x
        z.backward()
        np.testing.assert_allclose(z.data, 12.)
        np.testing.assert_allclose(x.grad, 7.)

    def test_generation_order_on_branching_graph(self):
        x = Variable(np.array(2.0))
        a = x * x
        z = a * a + a * a
        z.backward()
        np.testing.assert_allclose(x.grad, 64.)

    def test_same_function_appearing_in_multiple_paths(self):
        x = Variable(np.array(2.0))
        a = x * x
        z = a + a + a
        z.backward()
        np.testing.assert_allclose(x.grad, 12.)

    def test_multiple_trainable_operands_in_combined_expression(self):
        x = Variable(np.array(2.0))
        y = Variable(np.array(3.0))
        z = x / y + x * y - y
        z.backward()
        np.testing.assert_allclose(z.data, 2/3 + 6 - 3)
        np.testing.assert_allclose(x.grad, 1/3 + 3)
        np.testing.assert_allclose(y.grad, -2/9 + 2 - 1)

    def test_backward_uses_custom_output_gradient(self):
        x = Variable(np.array(2.0))
        y = x * x
        y.grad = np.array(3.0)
        y.backward()
        np.testing.assert_allclose(x.grad, 12.)

    def test_retain_intermediate_grad(self):
        for retain in (False, True):
            with self.subTest(retain_grad=retain):
                x = Variable(np.array(2.0))
                a = x * x
                y = a * a
                y.backward(retain_grad=retain)
                np.testing.assert_allclose(x.grad, 32.)
                if retain:
                    np.testing.assert_allclose(a.grad, 8.)
                    np.testing.assert_allclose(y.grad, 1.)
                else:
                    self.assertIsNone(a.grad)
                    self.assertIsNone(y.grad)

    def test_cleargrad_and_recompute(self):
        x = Variable(np.array(2.0))
        (x * x).backward()
        np.testing.assert_allclose(x.grad, 4.)
        x.cleargrad()
        self.assertIsNone(x.grad)
        (x * x).backward()
        np.testing.assert_allclose(x.grad, 4.)

    def test_function_output_is_weakly_referenced(self):
        f = Square()
        y = f(Variable(np.array(2.0)))
        y_ref = weakref.ref(y)
        self.assertIs(f.outputs[0](), y)
        del y
        gc.collect()
        self.assertIsNone(y_ref())
        self.assertIsNone(f.outputs[0]())

    def test_multi_output_both_used(self):
        x = Variable(np.array(2.0))
        outputs = Pair()(x)
        self.assertIsInstance(outputs, list)
        self.assertEqual(len(outputs), 2)
        a, b = outputs
        (a + b).backward()
        np.testing.assert_allclose(x.grad, 7.)

    def test_no_grad_disables_graph_but_keeps_forward(self):
        x = Variable(np.array(2.0))
        with no_grad():
            self.assertFalse(Config.enable_backprop)
            y = x * x + 1
            np.testing.assert_allclose(y.data, 5.)
            self.assertIsNone(y.creator)
        self.assertTrue(Config.enable_backprop)

    def test_nested_using_config_restores_previous_value(self):
        x = Variable(np.array(2.0))
        with no_grad():
            self.assertFalse(Config.enable_backprop)
            with using_config('enable_backprop', True):
                self.assertTrue(Config.enable_backprop)
                self.assertIsNotNone((x * x).creator)
            self.assertFalse(Config.enable_backprop)
        self.assertTrue(Config.enable_backprop)

    def test_no_grad_restored_after_exception(self):
        with self.assertRaisesRegex(RuntimeError, 'test exception'):
            with no_grad():
                self.assertFalse(Config.enable_backprop)
                raise RuntimeError('test exception')
        self.assertTrue(Config.enable_backprop)


class ExampleTests(unittest.TestCase):
    """书中的典型目标函数和简单数值微分校验。"""

    def test_matyas_gradient(self):
        x = Variable(np.array(1.0))
        y = Variable(np.array(2.0))
        z = .26 * (x ** 2 + y ** 2) - .48 * x * y
        z.backward()
        np.testing.assert_allclose(z.data, .34)
        np.testing.assert_allclose(x.grad, -.44)
        np.testing.assert_allclose(y.grad, .56)

    def test_goldstein_gradient_against_numerical_difference(self):
        def goldstein(x, y):
            a = 1 + (x + y + 1)**2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)
            b = 30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2)
            return a * b

        x, y = Variable(np.array(.7)), Variable(np.array(-.3))
        z = goldstein(x, y)
        z.backward()
        eps = 1e-6
        gx = (goldstein(.7 + eps, -.3) - goldstein(.7 - eps, -.3)) / (2*eps)
        gy = (goldstein(.7, -.3 + eps) - goldstein(.7, -.3 - eps)) / (2*eps)
        np.testing.assert_allclose(x.grad, gx, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(y.grad, gy, rtol=1e-5, atol=1e-5)


class KnownLimitationsTests(unittest.TestCase):
    """保留未来要实现的边界场景，但不要求当前教学版本通过。"""

    @unittest.expectedFailure
    def test_pending_multi_output_one_unused_but_live(self):
        x = Variable(np.array(2.0))
        a, b = Pair()(x)
        (2 * a).backward()
        np.testing.assert_allclose(x.grad, 8.)
        self.assertIsNone(b.grad)

    @unittest.expectedFailure
    def test_pending_multi_output_one_reclaimed(self):
        x = Variable(np.array(2.0))
        a, b = Pair()(x)
        del b
        gc.collect()
        (2 * a).backward()
        np.testing.assert_allclose(x.grad, 8.)

    @unittest.expectedFailure
    def test_pending_broadcast_add_gradient_reduction(self):
        x = Variable(np.ones((2, 3)))
        b = Variable(np.ones(3))
        (x + b).backward()
        np.testing.assert_allclose(b.grad, np.array([2., 2., 2.]))

    @unittest.expectedFailure
    def test_pending_broadcast_sub_gradient_reduction(self):
        x = Variable(np.ones((2, 3)))
        b = Variable(np.ones(3))
        (x - b).backward()
        np.testing.assert_allclose(b.grad, np.array([-2., -2., -2.]))

    @unittest.expectedFailure
    def test_pending_broadcast_mul_gradient_reduction(self):
        x = Variable(np.full((2, 3), 2.))
        b = Variable(np.ones(3))
        (x * b).backward()
        np.testing.assert_allclose(b.grad, np.array([4., 4., 4.]))

    @unittest.expectedFailure
    def test_pending_broadcast_div_gradient_reduction(self):
        x = Variable(np.full((2, 3), 2.))
        b = Variable(np.ones(3))
        (x / b).backward()
        np.testing.assert_allclose(b.grad, np.array([-4., -4., -4.]))


if __name__ == '__main__':
    unittest.main()
