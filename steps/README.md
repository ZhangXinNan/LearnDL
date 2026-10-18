# LearnDL — 书籍步骤代码（v0.1–v0.2）

本目录逐步记录《深度学习入门2：自制框架》的学习实现，不要求每个中间 Step 都提供完整、独立的框架。v0.1 对应 Step01–10，v0.2 对应 Step11–24；**Step05、Step15 是理论步骤**，因此没有对应 `.py` 文件。

## 环境

Python 3.11+、NumPy。可直接使用已有 Python/Conda 环境：

```bash
python -m pip install numpy
```

## 运行（在仓库根目录）

```bash
python steps/step09.py    # v0.1：简单计算链
python steps/step16.py    # v0.2：共享中间变量和计算图遍历
python steps/step22.py    # v0.2：加减乘除、负号、乘方
python steps/step23.py    # v0.2：通过 dezero 包使用 Variable
python steps/step24.py    # v0.2：复杂函数求导

python -m unittest discover -s steps -p 'step10.py' -v   # v0.1 测试
python -m unittest discover -s tests -p 'test_v02.py' -v # v0.2 测试
```

代码结构：`steps/` 保存每一阶段的教学过程，`dezero/core_simple.py` 保存当前整合实现，`dezero/__init__.py` 提供 `from dezero import Variable` 等入口。LearnDL 是项目名称；`dezero` 是当前 Python 包名，以便对照教材。

学习要点：[Step01–24](../docs/深度学习入门2自制框架/steps.md)；版本记录：[v0.1](../docs/v0.1.md)、[v0.2](../docs/v0.2.md)。
