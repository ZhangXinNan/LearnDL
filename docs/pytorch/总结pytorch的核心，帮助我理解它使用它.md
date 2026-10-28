## 一句话概括

PyTorch = **带自动求导的张量库（Tensor + Autograd）** + **搭模型的积木（nn.Module）** + **训练配套（优化器、数据加载）**。你刚看过 DeZero，PyTorch 的核心机制和它几乎一一对应，只是工业级、更完整。

| DeZero | PyTorch |
|---|---|
| Variable | `torch.Tensor`（`requires_grad=True`） |
| Function.forward/backward | `autograd.Function` / 内置算子 |
| Parameter / Layer / Model | `nn.Parameter` / `nn.Module` |
| Optimizer | `torch.optim` |
| Dataset / DataLoader | `torch.utils.data` |
| `no_grad()` | `torch.no_grad()` |
| `xp`（NumPy/CuPy 切换） | `.to(device)` |

## 1. Tensor：一切的基础

Tensor 就是带三个关键属性的多维数组：**shape、dtype、device**。

```python
import torch
x = torch.tensor([[1., 2.], [3., 4.]])
x.shape, x.dtype, x.device       # (2,2), float32, cpu

x = torch.zeros(2, 3); torch.randn(2, 3); torch.arange(6)
y = x.reshape(3, 2)              # 改形状
z = x.to("cuda")                 # 换设备（有GPU时）
n = x.numpy()                    # 与NumPy互转（CPU上共享内存）
```

要点：

- **广播**规则与 NumPy 相同，几乎所有 shape 错误都出在这里。
- `view` / 切片 / `transpose` 通常**共享内存**不复制，修改一个会影响另一个；要独立副本用 `.clone()`。
- **dtype 有讲究**：模型参数默认 `float32`；分类标签要 `int64`（`long`）。

## 2. Autograd：自动求导

和 DeZero 一样是**动态图**：前向时建图，`backward()` 时反向走一遍。

```python
x = torch.tensor(3.0, requires_grad=True)
y = x * x + x
y.backward()
print(x.grad)        # tensor(7.)
```

每个计算结果带 `grad_fn`，指向产生它的运算，这就是 DeZero 的 `creator`。

必须记住的行为：

- **梯度默认累加**（DeZero 里的 `cleargrad` 就是为此），所以每个 batch 要先清零：`optimizer.zero_grad()`。
- `with torch.no_grad():` 里不建图，用于推理和评估。
- `tensor.detach()` 切断与图的联系，常用于取出数值或阻止梯度回传。
- 中间变量默认不保留梯度（对应 DeZero 的 `retain_grad`），需要时调用 `.retain_grad()`。
- 高阶导数：`torch.autograd.grad(y, x, create_graph=True)`。
- 反向传播后计算图默认被释放，想再次 backward 要设 `retain_graph=True`。

## 3. nn.Module：搭模型

继承 `nn.Module`，在 `__init__` 里定义层，在 `forward` 里写前向：

```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 10),
        )
    def forward(self, x):
        return self.net(x)       # 用 model(x)，不要直接调 forward
```

和 DeZero 的 `Layer` 一样，赋给 `self` 的子模块和 `nn.Parameter` 会被自动登记，所以：

- `model.parameters()` 能收集全部参数交给优化器
- `model.state_dict()` 能导出所有权重
- `model.to(device)` 一次性搬迁所有参数

常用层：`Linear`、`Conv2d`、`BatchNorm`、`Dropout`、`Embedding`、`LSTM`、`TransformerEncoderLayer`。常用损失：`CrossEntropyLoss`（分类）、`MSELoss`（回归）、`BCEWithLogitsLoss`（二分类）。

## 4. 标准训练流程

几乎所有 PyTorch 训练代码都是这个骨架：

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model = MLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(num_epochs):
    model.train()                          # 训练模式
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()              # 1. 清梯度
        loss = criterion(model(x), y)      # 2. 前向+损失
        loss.backward()                    # 3. 反向
        optimizer.step()                   # 4. 更新参数

    model.eval()                           # 评估模式
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            ...
```

记住 **清零 → 前向 → 反向 → 更新** 这四步，剩下都是在此基础上加东西（学习率调度、梯度裁剪、混合精度等）。

## 5. 数据加载

```python
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, xs, ys): self.xs, self.ys = xs, ys
    def __len__(self): return len(self.xs)
    def __getitem__(self, i): return self.xs[i], self.ys[i]

loader = DataLoader(MyDataset(xs, ys), batch_size=64, shuffle=True, num_workers=2)
```

只需实现 `__len__` 和 `__getitem__`，`DataLoader` 负责分批、打乱、多进程读取。图像任务常配合 `torchvision.transforms` 做预处理和数据增强。

## 6. 保存与加载

```python
torch.save(model.state_dict(), "model.pt")        # 推荐：只存权重

model = MLP()
model.load_state_dict(torch.load("model.pt", map_location=device))
model.eval()
```

要断点续训，把 `optimizer.state_dict()`、epoch 等一起存进一个字典。

## 7. 最常见的坑

1. **忘记 `zero_grad()`**：梯度越积越大，loss 表现怪异。
2. **忘记 `model.eval()` + `no_grad()`**：Dropout 和 BatchNorm 行为不对，评估结果不稳定，还浪费显存。
3. **设备不一致**：模型和数据必须在同一设备，报错信息里会出现 `cuda` 和 `cpu` 混用。
4. **`CrossEntropyLoss` 已含 softmax**：模型最后一层输出 logits，**不要**再加 softmax。标签是类别索引（`long`），不是 one-hot。
5. **记录 loss 用 `loss.item()`**：直接累加 `loss` 会一直持有计算图，显存持续增长。
6. **原地操作（`x += 1`、`relu_`）** 可能破坏求导所需的中间值，报 "modified by an inplace operation"，换成非原地写法即可。
7. **shape 不对**：调试时多 `print(x.shape)`；批维度在最前面（`[B, C, H, W]`）。

## 8. 进阶工具（先知道有，需要再学）

- **`torch.compile(model)`**：2.x 的编译加速，一行代码，常能提速。
- **混合精度 `torch.autocast`**：用 fp16/bf16 提速省显存。
- **`DistributedDataParallel`**：多卡训练，与你学的分布式系统课直接相关。
- **`torch.profiler`**：定位性能瓶颈。
- **Hugging Face、Lightning** 等库构建于 PyTorch 之上，理解了本文就能读懂它们的代码。

## 建议的学习路径

1. 用上面的训练骨架，在 MNIST 上跑通一个 MLP
2. 换成 CNN，加上数据增强，体会 `train/eval` 的差别
3. 手写一次训练循环，不用任何封装；再读官方教程 *Learning PyTorch with Examples*，它从 NumPy 一步步演进到 `nn.Module`，和 DeZero 的学习路线很像
4. 读一个真实项目的 `train.py`，对照本文骨架找出各部分

需要的话，我可以把这些整理成一份可保存的文档，或者带你一起写一个完整的 MNIST 训练脚本，逐行解释。



