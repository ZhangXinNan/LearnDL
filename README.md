# LearnDL

A personal, long-term project for understanding the **fundamental mechanisms of deep learning** through mathematical reasoning, small implementations, and verification.

The emphasis is on understanding how fundamental operations work (such as backpropagation and SGD) rather than building a production-grade framework. Higher-level PyTorch models belong to separate projects such as LearnOCR.

## Documents

| English | 中文 |
| --- | --- |
| [Goals](docs/goals.md) | [目标要求](docs/目标要求.md) |
| [Workflow](docs/workflow.md) | [开发流程](docs/开发流程.md) |
| [Versioning](docs/versioning.md) | [版本规范](docs/版本规范.md) |

The **v0.6 release candidate** covers textbook Steps 01–51 (automatic differentiation, tensor operations, neural networks, and classification). The project is named LearnDL; the importable Python package is `dezero` to match the textbook. The v0.6 release is not finalized until the real MNIST CPU acceptance run has been recorded.

## Quick start

Run from the repository root:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m steps.step50   # Spiral: 300 epochs, real held-out test data
python -m steps.step51   # MNIST: downloads data on first run; CPU training may take time
```

Learning notes: [Steps](docs/BuildingDeepLearningFramwork/steps.md) · [v0.1](docs/v0.1.md) · [v0.2](docs/v0.2.md) · [v0.6 review](docs/v0.6-review.md) · [Development plan](docs/development-plan.md).
