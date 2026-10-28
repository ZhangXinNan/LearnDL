# LearnDL — 教材步骤（Step01–Step51）

本目录按《深度学习入门2：自制框架》的顺序记录学习代码。Step05、Step15 为理论内容，没有独立的 `.py` 文件。中间步骤仅用于说明当时的原理；最终整合版在 `dezero/` 中。

## 运行

在**仓库根目录**执行：

```bash
python -m pip install -r requirements.txt
python -m steps.step46   # MLP + SGD 回归
python -m steps.step48   # Spiral 分类
python -m steps.step50   # Spiral + DataLoader + 独立测试集
python -m steps.step51   # MNIST：首次下载、CPU 训练
python -m unittest discover -s tests -v
```

也可以在 IDE 中将项目根目录加入 `PYTHONPATH`。直接执行 `python steps/step50.py` 时，若运行环境未添加项目根目录，可能找不到 `dezero` 包。

- v0.1：Step01–10，自动微分基础。
- v0.2：Step11–24，复杂计算图与运算符。
- v0.3：Step25–36，高阶自动微分。
- v0.4：Step37–41，张量运算与梯度。
- v0.5：Step42–46，神经网络和 SGD。
- v0.6：Step47–51，分类损失、Dataset、DataLoader、MNIST。

学习要点：[Steps 学习笔记](../docs/BuildingDeepLearningFramwork/steps.md)；[v0.6 审查与验收记录](../docs/v0.6-review.md)。
