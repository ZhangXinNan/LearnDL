
import numpy as np
from dezero import Variable, Model
from dezero import optimizers
import dezero.functions as F
import dezero.layers as L
from dezero.models import MLP

# 数据集
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

# 设置超参数
lr = 0.2
max_iter = 10000
hidden_size = 10


model = MLP((hidden_size, 1))
optimizer = optimizers.SGD(lr)
optimizer.setup(model)

# 3 神经网络的训练
for i in range(max_iter):
    y_pred = model(x)
    loss = F.mean_squared_error(y, y_pred)
    model.cleargrads()
    loss.backward()

    optimizer.update()
    if i % 1000 == 0:
        print(i, loss)




