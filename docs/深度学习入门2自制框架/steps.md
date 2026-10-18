# LearnDL 逐步学习要点（Step 01–24）

> 参考：《深度学习入门2：自制框架》  
> 记录范围：Step 01–24（Step 05、15 以理论为主，没有对应的 `.py` 文件）。  
> 用法：**每完成一步，理解它解决了上一版的什么问题，再继续下一步**。后续步骤实现后继续追加，不提前把教学代码改造成完整框架。

## 一条主线

`Variable 装数据 → Function 做运算 → 连续调用形成计算链 → 数值微分 → 链式法则 → 手动反传 → 自动记录计算图 → 循环反传 → 易用接口 → 测试 → 多输入输出 → 梯度累积 → 正确的计算图顺序 → 弱引用和内存管理 → 运算符重载 → 封装 `dezero` → 复杂函数求导`

## v0.1：简单计算链的自动微分（Step 01–10）

| 步骤 | 新增内容 | 需要掌握的要点 |
| --- | --- | --- |
| [01：变量](../../steps/step01.py) | 用 `Variable` 包装 NumPy 数组。 | **变量与数据不是同一回事**：`Variable` 是容器，`data` 是实际数值。为什么不直接传 `ndarray`？ |
| [02：函数](../../steps/step02.py) | `Function.__call__()`、`forward()`、`Square`。 | 理解一次前向计算：取出输入 `data` → 计算 → 包装新的 `Variable`。为什么把通用调用与具体运算分开？ |
| [03：连续调用](../../steps/step03.py) | 增加 `Exp`，组合 `Square → Exp → Square`。 | 多个简单算子如何表示复合函数；中间的每个 `Variable` 如何承接上一层输出。 |
| [04：数值微分](../../steps/step04.py) | 中心差分 `numerical_diff()`。 | 用 `(f(x+ε)-f(x-ε))/(2ε)` 近似导数；知道它可用来**检查梯度**，但有误差、计算成本高。 |
| **05：反向传播理论** | 无单独代码文件，先学习链式法则。 | 对复合函数，**最终梯度 = 沿计算链逐层相乘的局部导数**；理解反向传播为什么从输出开始。此知识在 Step 06 用到。 |
| [06：手动反传](../../steps/step06.py) | `grad`、各算子的 `backward()`，逐层手工调用。 | `Square` 的局部导数是 `2x`，`Exp` 是 `exp(x)`；`gy` 从后往前传，`gx` 是传给前一层的梯度。 |
| [07：自动反传](../../steps/step07.py) | `creator`、`set_creator()`、递归 `Variable.backward()`。 | 为什么每个输出需要记住**由哪个函数生成**；怎样沿 `creator` 自动找到前一步。 |
| [08：从递归到循环](../../steps/step08.py) | 用 `funcs` 列表和 `while` 反向遍历。 | 列表如何暂存待处理的函数；为什么循环更便于后续控制反传顺序。此时仅保证简单链正确。 |
| [09：更易使用](../../steps/step09.py) | `square()`、`exp()` 包装函数、`as_array()`、自动初始化输出梯度。 | 为什么 `y.backward()` 能省去手动设 `y.grad=1`；为何要求 `data` 为 `ndarray`，并把标量输出转换成数组。 |
| [10：测试](../../steps/step10.py) | `unittest`：前向、反向、数值梯度、复合函数检查。 | **能运行不等于导数正确**：用解析导数和中心差分交叉验证。学会小规模、可重复的梯度测试。 |

**阅读顺序是 01 → 02 → 03 → 04 → 05（理论）→ 06 → … → 10。** Step 05 没有 `.py` 文件，并非漏写。

## v0.2：复杂计算图与自然表达式（Step 11–24）

| 步骤 | 新增内容 | 需要掌握的要点 |
| --- | --- | --- |
| [11：可变长输入·初版](../../steps/step11.py) | `Function` 改为接收输入列表、返回输出列表，`Add` 接收两个输入。 | 从 `input/output` 变为 `inputs/outputs`；理解为什么多输入算子不能再沿用单输入字段。**本步只演示前向**。 |
| [12：可变长输入·改进](../../steps/step12.py) | `*inputs`、`self.forward(*xs)`，单输出直接返回 `Variable`，封装 `add(x0, x1)`。 | 看懂 Python 的**位置参数收集与解包**；为什么使用者写 `add(x0, x1)` 比写列表方便。**仍主要演示前向**。 |
| [13：多输入反向传播](../../steps/step13.py) | 用 `gys = [output.grad ...]`、`f.backward(*gys)`，并把 `gxs` 分发到各输入；实现 `Add.backward()`。 | **一个输出梯度怎样变成多个输入梯度**？对 `y=x0+x1`，两个输入的梯度均为 `gy`。当前尚未累积重复输入的梯度。 |
| [14：重复使用变量](../../steps/step14.py) | 当 `x.grad` 已存在时累加 `gx`；增加 `cleargrad()`。 | 为什么 `y=add(x,x)` 对 `x` 的梯度应为 2 而不是 1；为什么多次计算前需要按需清空梯度。**累积梯度还不等于支持所有复杂分支图**。 |

### Step 14 的一个关键边界（将在 Step 16 解决）

当前代码对 `y = add(x, x)` 能求出正确的梯度 `2`；但如果同一个**中间结果**被重复使用：

```python
x = Variable(np.array(3.0))
a = add(x, x)
y = add(a, a)
y.backward()
# 正确的 x.grad 应为 4；当前 Step 14 会得到 8。
```

原因是反向传播时同一个 `a.creator` 可能被放入 `funcs` 多次，造成重复处理。这是 **Step 15（理解复杂图的反传顺序）和 Step 16（用 generation 和去重实现正确顺序）** 要解决的问题，不必在 Step 14 提前改完。

此外，Step 11、12 临时沿用了 Step 09 的 `Variable`，其旧版 `backward()` 尚不兼容新的多输入 `Function`。因此这两步只用于观察前向接口演变；从 Step 13 开始再验证多输入反向传播。多输出中只有部分输出参与后续计算等边界，也留待后续完善。

## v0.2 后半段：复杂计算图、内存管理与易用接口（Step 15–24）

| 步骤 | 新增内容 | 需要掌握的要点 |
| --- | --- | --- |
| **15：复杂计算图（理论）** | 书籍理论步骤，无 `step15.py`。 | **重复引用中间变量与重复使用叶子变量不是一回事**：反传既要累积多个分支的梯度，又不能把同一个 `Function` 错误执行多次；理解为什么必须先处理后代节点。 |
| [16：复杂计算图（实现）](../../steps/step16.py) | `generation`、`seen_set` 和按层级排序的待处理函数列表。 | 为什么函数只加入一次，以及为什么总是从计算图更靠后的函数开始反传；能解释 `a=x²; y=a²+a²` 的正确梯度。 |
| [17：内存管理与循环引用](../../steps/step17.py) | 让 `Function.outputs` 保存 `weakref.ref`。 | 区分普通强引用和弱引用；为什么 `output()` 要解引用；理解不需要为了存计算图而永久持有输出变量。 |
| [18：节省内存的模式](../../steps/step18.py) | `retain_grad`、`Config.enable_backprop`、`using_config()`、`no_grad()`。 | 叶子梯度与中间变量梯度的保留需求不同；推理时为什么无需建立计算图；`try/finally` 为什么保证配置恢复。 |
| [19：让 Variable 更易用](../../steps/step19.py) | 名称、`shape`、`ndim`、`size`、`dtype`、`__len__`、`__repr__`。 | 这些属性只是透传底层 NumPy 数据的元信息，不改变自动微分核心；明确零维数组不能直接 `len()`。 |
| [20：运算符重载（1）](../../steps/step20.py) | `Mul` 及 `__mul__`、`__add__`。 | Python 怎样把 `a * b + c` 转换为 `Function` 调用；乘法为什么给两个输入传回不同的局部梯度。 |
| [21：运算符重载（2）](../../steps/step21.py) | `as_variable()`、`__array_priority__`、反向加法/乘法。 | 为什么 `3 + x` 与 `x + 3` 都要支持；Python 标量怎样经过 `as_array()` 转为 `Variable`；能检查前向值也检查反向梯度。 |
| [22：运算符重载（3）](../../steps/step22.py) | `Neg`、`Sub`、`Div`、`Pow` 及反向运算。 | 逐个写出加减乘除和常数幂的局部求导公式；`3-x` 和 `3/x` 为什么必须交换输入顺序。 |
| [23：打包](../../steps/step23.py) | 创建 `dezero` 包、从 `dezero` 导入 `Variable`，并集中调用 `setup_variable()`。 | 区分独立 Step 教学文件和可以导入的公共包；`__init__.py` 是对外入口；当前正式包名是 `dezero`，项目名称仍是 LearnDL。 |
| [24：复杂函数求导](../../steps/step24.py) | Sphere、Matyas、Goldstein–Price 函数实例。 | 大型表达式无需手写整体导数，自动微分通过基本算子组合求梯度；用解析梯度或中心差分独立校验结果。 |

**本版核心例子：** 对 `x = 2`，`a = x*x`，`z = a*a + a*a`，应有 `z = 32`、`dz/dx = 64`；对 Matyas 函数 `f(x,y)=0.26(x²+y²)-0.48xy`，在 `(1,2)` 处梯度为 `(-0.44, 0.56)`。

**已知边界：** 当前通用算子虽可在 NumPy 数组上前向广播，但广播产生的梯度尚未归约到输入形状；这项能力安排在 v0.4。`retain_grad=False` 还会清除输出变量自身的 `.grad`，使用时不要误以为它被永久保存。

## 运行与自检

在项目根目录 `LearnDL/` 执行：

```bash
python steps/step09.py          # 单链自动微分
python -m unittest discover -s steps -p 'step10.py' -v
python steps/step13.py          # 多输入反向传播
python steps/step14.py          # 同一变量重复使用、cleargrad
python steps/step16.py          # 分支图的梯度与 generation
python steps/step24.py          # Sphere、Matyas、Goldstein–Price
python -m unittest discover -s tests -p "test_v02.py" -v
```

以下六个问题，能不用看代码解释清楚，就可以继续：

1. 为什么 Step 07 给 `Variable` 增加 `creator`，而 Step 08 又把递归改成循环？
2. Step 11–13 怎样将原来一个输入、一个梯度的接口，改成多个输入和多路梯度？
3. Step 14 为什么要**累积梯度**；为什么 `x+x` 正确还不足以说明复杂计算图正确？
4. 为什么 Step 16 的 `generation` 和去重需要同时存在？
5. 为什么 `no_grad()` 可以省内存？弱引用为何仍允许输出被垃圾回收？
6. `3-x` 的反向传播与 `x-3` 的反向传播有什么不同？

**后续维护：** 完成新的 Step 后，只追加对应的一行“新增内容 + 需要掌握的要点”。版本范围详见 [开发计划](../development-plan.md)；版本总结见 [v0.1](../v0.1.md) 和 [v0.2](../v0.2.md)。
