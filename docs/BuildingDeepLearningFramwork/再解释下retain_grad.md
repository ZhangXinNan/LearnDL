`retain_grad` 是 DeZero 自动微分框架中一个非常重要、但原理其实很简单的参数。

先记住一句话：

> `retain_grad` 决定反向传播完成后，是否保留中间变量的梯度。它不决定是否计算梯度，而是决定计算完之后要不要清除。

结合你现在的 LearnDL `core_simple.py`，我们从一个简单例子理解。

## 一、先看一个例子

```
import numpy as npfrom dezero import Variablex = Variable(np.array(2.0))a = x * xy = a * ay.backward()
```

它对应的数学表达式为：

$$
a=x^2,\qquad y=a^2=x^4
$$

当 $x=2$ 时：

$$
x=2,\quad a=4,\quad y=16
$$

反向传播的梯度为：

$$
\frac{\partial y}{\partial a}=2a=8
$$

$$
\frac{\partial y}{\partial x} =\frac{\partial y}{\partial a}\frac{\partial a}{\partial x} =8\times4=32
$$

计算图与梯度

上图显示反向传播过程中计算出的梯度，暂未考虑是否清除。

现在问题来了：既然梯度都算出来了，为什么还要清除一部分？

## 二、retain_grad=False（默认）

```
y.backward(retain_grad=False)
```

反向传播过程中：

1. 计算 `y.grad = 1`。
2. 经过最后一个乘法节点，计算出 `a.grad = 8`。
3. `y.grad` 已经使用完毕，清除它。
4. 继续反向传播，通过第一个乘法节点计算出 `x.grad = 32`。
5. `a.grad` 已经使用完毕，清除它。

最终：

| 变量      | data | grad   |
| ------- | ---- | ------ |
| x（叶子变量） | 2    | 32     |
| a（中间变量） | 4    | `None` |
| y（最终输出） | 16   | `None` |

为什么保留 `x.grad`？

因为 `x` 是叶子变量，它不是任何 `Function` 的输出，所以不会被这段清理代码清除。

在实际模型训练中，我们一般主要关心参数变量的梯度，例如权重、偏置，而不是所有中间计算结果的梯度。

## 三、retain_grad=True

```
y.backward(retain_grad=True)
```

计算过程完全相同，只是不清理已经使用完的梯度。

| 变量      | data | grad |
| ------- | ---- | ---- |
| x（叶子变量） | 2    | 32   |
| a（中间变量） | 4    | 8    |
| y（最终输出） | 16   | 1    |

交互演示：切换 retain_grad

False（默认）True

x

## 32

grad叶子变量

a

## None

grad中间变量

y

## None

grad最终输出

清除已使用的中间梯度，只保留叶子变量 x 的梯度，减少内存占用。

## 四、回到你自己的 core_simple.py

你现在的代码：

```
def backward(self, retain_grad=False):    ...    while funcs:        f = funcs.pop()        gys = [output().grad for output in f.outputs]        gxs = f.backward(*gys)        ...        if not retain_grad:            for y in f.outputs:                y().grad = None
```

关键就是最后三行：

```
if not retain_grad:    for y in f.outputs:        y().grad = None
```

逐行理解：

第一行：

```
if not retain_grad:
```

只有 `retain_grad=False` 时才执行梯度清理。

第二行：

```
for y in f.outputs:
```

遍历当前 `Function` 的所有输出变量。之所以是 `outputs`，是因为这些输出的梯度已经在当前函数的 `backward()` 中使用完了。

第三行：

```
y().grad = None
```

将输出变量的梯度清除。

这里的 `y()` 正是前面学过的 `weakref`：通过弱引用取得对应的 `Variable`。

注意，清除的是 `Variable.grad`，不是 `Variable.data`，也不是 `Variable.creator`。

因此：

- `a.data` 仍然是 4。
- `a.creator` 仍然指向创建它的 Function。
- `a.grad` 被设置为 `None`。

这也说明 `retain_grad=False` 并不是删除计算图，更不是关闭反向传播。

## 五、自己运行对比一下

```
import numpy as np
from dezero import Variable

def test(retain_grad):
    x = Variable(np.array(2.0))
    a = x * x
    y = a * a
    y.backward(retain_grad=retain_grad)
    print(f"retain_grad = {retain_grad}")
    print("x.grad =", x.grad)
    print("a.grad =", a.grad)
    print("y.grad =", y.grad)


test(False)
test(True)
```

输出：

```
retain_grad = False
x.grad = 32.0
a.grad = None
y.grad = None

retain_grad = True
x.grad = 32.0
a.grad = 8.0
y.grad = 1.0
```

## 六、不要与另外两个概念混淆

| 功能                 | 作用             |
| ------------------ | -------------- |
| `no_grad()`        | 前向计算时不建立计算图    |
| `retain_grad=True` | 反向传播时保留中间变量的梯度 |
| `cleargrad()`      | 手动清除某个变量的梯度    |

还有一个容易混淆的地方：PyTorch 中的 `retain_graph=True` 与 DeZero 的 `retain_grad=True` 不是同一个概念。

`retain_graph` 主要用于控制反向传播后是否保留继续求导所需的计算图资源，而你当前 LearnDL 的 `retain_grad` 只控制中间梯度是否清除。

## 七、最后用一句话真正理解它

你可以把反向传播想象成一条流水线：

每个 Function 收到输出梯度 → 计算输入梯度 → 将输入梯度传递给前面的节点。

输出梯度完成了自己的任务，就可以被释放。

- `retain_grad=False`：用完就丢，节省内存。
- `retain_grad=True`：用完保留，方便观察和调试。

对于 LearnDL 来说，我建议你平时保持默认 `False`，但在研究计算图和链式求导时使用 `True`，观察每个中间变量的梯度。

真正要理解的关键是：反向传播必须计算中间梯度，但并不需要在计算结束后永久保存它们。



