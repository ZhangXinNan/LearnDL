`create_graph` 是 LearnDL 第三阶段（v0.3）最重要的概念之一，也是理解\*\*高阶自动微分（Higher-Order Automatic Differentiation）\*\*的关键。

先记住一句话：

> `create_graph=True` 的作用是：在反向传播计算梯度时，把计算梯度的过程也记录成一张新的计算图，使梯度本身能够继续求导。

这里最容易困惑的是：不是已经有计算图了吗？为什么反向传播还要再创建一张图？

我们结合你现在的 `dezero/core.py` 来理解。

## 一、为什么需要 create_graph？

先看一个简单函数：

$$
y=x^4
$$

它的一阶和二阶导数分别为：

$$
\frac{dy}{dx}=4x^3
$$

$$
\frac{d^2y}{dx^2}=12x^2
$$

当 $x=2$ 时：

| 项目             | 结果 |
| -------------- | -- |
| 原函数 $y$    | 16 |
| 一阶导数 $y'$  | 32 |
| 二阶导数 $y''$ | 48 |

我们在 LearnDL 中运行：

```
import numpy as np
from dezero import Variable
x = Variable(np.array(2.0))
y = x ** 4
y.backward()
print(x.grad)  # variable(32.0)
```

这时成功计算出一阶导数。

但如果我们希望继续计算二阶导数呢？

```
gx = x.grad
x.cleargrad()
gx.backward()
print(x.grad)  # None
```

为什么得到 `None`？

因为默认执行：

```
y.backward(create_graph=False)
```

在第一次反向传播时，虽然计算了：

$$
g_x=4x^3
$$

却没有记录计算 $4x^3$ 的过程。

也就是说，`gx` 虽然是 `Variable`，但已经失去了对原始变量 `x` 的计算图依赖。

因此，无法从 `gx` 继续反向传播到 `x`。

## 二、create_graph=True 如何解决？

只需要修改一个参数：

```python
x = Variable(np.array(2.0))
y = x ** 4
# 第一次反向传播：建立梯度计算图
y.backward(create_graph=True)
gx = x.grad
print(gx)
# variable(32.0)
# 清除之前的一阶梯度
x.cleargrad()
# 第二次反向传播
gx.backward()
print(x.grad)  # variable(48.0)
```

这里其实经历了两次反向传播。

第一次

原函数的计算图


前向传播：计算 y = x⁴；反向传播：计算 dy/dx = 32。

第二次

梯度本身的计算图

create_graph=True 记录了计算 gx = 4x³ 的运算，因此还能沿新图对 x 求导，得到 48。

注意：上图的第二张图是概念示意。你的实际 LearnDL 会把 `4 * x**3 * gy` 分解成多个 `Function` 节点，而不是一个节点。

最重要的区别是：第一张图计算原函数，第二张图记录导数的计算过程。

## 三、回到你自己的 core.py

这是整个 `create_graph` 实现最关键的地方。

你目前的代码是：

```python
def backward(self, retain_grad=False, create_graph=False):
    #...    
    while funcs:
        f = funcs.pop()
        gys = [output().grad for output in f.outputs]
        with using_config('enable_backprop', create_graph):
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)
            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx
                if x.creator is not None:
                    add_func(x.creator)
```

重点是这一句：

```python
with using_config('enable_backprop', create_graph):
```

`using_config()` 实际就是临时修改一个全局配置：

```python
Config.enable_backprop = create_graph
```

并在退出 `with` 时恢复原来的值。

而你的 `Function.__call__()` 中有：

```python
if Config.enable_backprop:
    self.generation = max([x.generation for x in inputs])
    for output in outputs:
        output.set_creator(self)
    self.inputs = inputs
    self.outputs = [weakref.ref(output) for output in outputs]
```

也就是说：

交互演示：create_graph 如何控制计算图

create_graph=Falsecreate_graph=True

反向传播内部

with using_config('enable_backprop', true):

gx = 4 \* x \*\* 3 \* gy

gx.creator = Mul 节点

一阶导数

# 32

两种情况都正确

二阶导数

# 48

可继续反向传播

所以实现原理其实很巧妙：

你没有另外编写一套高阶求导算法，而是复用了已有的 Function 和 Variable。

只要反向传播中的运算也被记录到计算图里，原来的 `backward()` 就能够再次使用。

这就是 DeZero 动态计算图设计的精妙之处。

## 四、为什么你的 Pow\.backward() 必须使用 Variable？

这是你第三阶段刚刚修复过的一个重要问题。

原来：

```python
def backward(self, gy):
    x = self.inputs[0].data
    c = self.c
    gx = c * x ** (c - 1) * gy
    return gx
```

现在：

```python
def backward(self, gy):
    x = self.inputs[0]
    c = self.c
    gx = c * x ** (c - 1) * gy
    return gx
```

两者最大的区别：

| 写法                    | 参与计算的类型         | 能否保留对原变量的求导关系 |
| --------------------- | --------------- | ------------- |
| `self.inputs[0].data` | `numpy.ndarray` | 不能            |
| `self.inputs[0]`      | `Variable`      | 能（需要启用建图）     |

尤其要注意：

`create_graph=True` 不是魔法。

它只能记录实际通过 `Function` 执行的运算。

如果在 `backward()` 中提前把输入转换成 `.data`，就丢失了对原始 `Variable` 的依赖关系，即使开启 `create_graph=True` 也无济于事。

这也是你之前 `x**4` 能算出一阶导数，却不能正确计算二阶导数的根本原因。

## 五、create_graph、retain_grad、no_grad 的区别

这三个功能经常容易混淆。

| 功能                  | 控制的内容            | 主要用途      |
| ------------------- | ---------------- | --------- |
| `create_graph=True` | 是否记录反向传播中执行的可微运算 | 高阶求导      |
| `retain_grad=True`  | 是否保留中间变量的梯度      | 观察和调试     |
| `no_grad()`         | 是否记录上下文中的计算操作    | 推理、减少内存消耗 |

举例：

```python
# 需要二阶导数
y.backward(create_graph=True)
# 只想观察中间变量的梯度
y.backward(retain_grad=True)
# 同时需要高阶导数和中间梯度
y.backward(retain_grad=True, create_graph=True)
```

尤其要记住：

`create_graph=True` 不代表 `retain_grad=True`。

前者决定梯度是否具有可继续求导的计算图，后者决定中间梯度计算完后是否清除。这是两个独立的机制。

## 六、再往深处想一步：为什么能够无限次求导？

比如：

$$
y=x^4
$$

我们可以依次计算：

$$
y'=4x^3
$$

$$
y''=12x^2
$$

$$
y'''=24x
$$

$$
y''''=24
$$

在你的 LearnDL 中：

```
import numpy as np
from dezero import Variable
x = Variable(np.array(2.0))
y = x ** 4
for i in range(4):
    x.cleargrad()
    y.backward(create_graph=True)
    print(f"{i + 1} 阶导数:", x.grad)
    y = x.grad
```

输出：

```
1 阶导数: variable(32.0)
2 阶导数: variable(48.0)
3 阶导数: variable(48.0)
4 阶导数: variable(24.0)
```

注意，这里当 $x=2$ 时，三阶导数 $24x=48$，所以二阶与三阶导数恰好相等。

这种写法反复执行：

Variable（当前函数）

backward(create_graph=True)

新的 Variable（当前导数）

新导数又能继续调用 backward()

理论上可以不断重复这个过程，但并不意味着所有函数都能无限次求导，也不意味着内存开销可以忽略。实际能求到多少阶，取决于函数的数学性质以及框架所支持的求导运算。

## 七、我建议你做一个小实验

在 PyCharm 中运行：

```python
import numpy as np
from dezero import Variable

for create_graph in [False, True]:
    x = Variable(np.array(2.0))
    y = x ** 4
    y.backward(create_graph=create_graph)
    gx = x.grad
    print("create_graph:", create_graph)
    print("gx:", gx)
    print("gx.creator:", gx.creator)
    x.cleargrad()
    gx.backward()
    print("second grad:", x.grad)
    print("-" * 30)
```

我用你最新上传的 LearnDL 代码验证了这段逻辑。

`False` 时：

```
create_graph: False
gx: variable(32.0)
gx.creator: None
second grad: None
```

`True` 时：

```
create_graph: True
gx: variable(32.0)
gx.creator: <dezero.core.Mul object at ...>
second grad: variable(48.0)
```

请重点观察这句：

```
print(gx.creator)
```

两个 `gx` 的数值完全相同，区别却在于有没有 `creator`。

这是理解 `create_graph` 最直观的切入点。

## 八、最后总结

如果把你目前学到的内容串起来：

- `Variable.data`：保存数值。
- `Variable.creator`：记录变量从哪个函数产生。
- `Function.inputs`：保留计算所依赖的输入。
- `backward()`：沿计算图反向传播梯度。
- `retain_grad`：决定中间梯度是否保留。
- `create_graph`：决定梯度计算过程是否也构成计算图。

其中 `create_graph` 最核心的思想只有一个：

> 求导本身也是一种计算。既然是计算，就可以建立计算图；既然能够建立计算图，就可以继续求导。

你现在已经实现的并不只是一个能够计算梯度的小工具，而是一个支持高阶自动微分的简化框架。

第三阶段真正值得理解的，就是这一层思想：自动微分不需要为二阶、三阶、四阶导数分别设计不同算法；只需要让第一次求导得到的计算过程仍然可微。


