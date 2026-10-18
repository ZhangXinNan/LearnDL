# LearnDL

A personal, long-term project for understanding the **fundamental mechanisms of deep learning** through mathematical reasoning, small implementations, and verification.

The emphasis is on understanding how fundamental operations work (such as backpropagation, SGD/Adam, and Batch Normalization), rather than building a production-grade framework or reproducing entire PyTorch models. Mainstream PyTorch model implementations belong to separate projects such as LearnOCR.

## Documents

| English | 中文 |
| --- | --- |
| [Goals](docs/goals.md) | [目标要求](docs/目标要求.md) |
| [Workflow](docs/workflow.md) | [开发流程](docs/开发流程.md) |
| [Versioning](docs/versioning.md) | [版本规范](docs/版本规范.md) |

`v0.0` is the documentation and project initialization baseline. The current code-freeze candidate is **v0.2** (complex computation graphs and basic arithmetic). The project is named LearnDL; the importable Python package is `dezero`, matching the textbook's naming.

## Quick start (v0.2)

From the repository root, with NumPy installed:

```bash
python -c "import numpy as np; from dezero import Variable; x = Variable(np.array(2.0)); y = x*x+x; y.backward(); print(y.data, x.grad)"  # 6.0 5.0
python -m unittest discover -s tests -p 'test_v02.py' -v
python -m unittest discover -s steps -p 'step10.py' -v
```

Learning notes: [Steps 01–24](docs/深度学习入门2自制框架/steps.md) · [v0.1](docs/v0.1.md) · [v0.2](docs/v0.2.md) · [Development plan](docs/development-plan.md).
