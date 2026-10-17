# LearnDL — 书籍步骤代码（v0.1）

本目录按《深度学习入门2：自制框架》的步骤保留学习时的独立示例。v0.1 对应步骤 1–10；步骤 5 以理论讲解为主，因此没有 `step05.py`。

## 运行环境

需要 Python 3.11+ 和 NumPy。可沿用已有环境；如需新建 Conda 环境：

```bash
conda create -n dezero python=3.11
conda activate dezero
pip install numpy
```

## 运行示例与测试

**以下命令均在项目根目录 `LearnDL/` 执行：**

```bash
# 查看 v0.1 的最终复合函数自动微分示例
python steps/step09.py

# 运行 v0.1 的 5 个单元测试（含复合函数的梯度检查）
python -m unittest discover -s steps -p 'step10.py' -v
```

也可以用 `python steps/step10.py` 直接运行这 5 个测试。

说明：`step01.py` 至 `step10.py` 是逐步演进的学习代码，不要求提前整理为公共包；正式的 `learndl` 包计划在 v0.2 引入。

学习记录见 [`docs/v0.1.md`](../docs/v0.1.md)。
