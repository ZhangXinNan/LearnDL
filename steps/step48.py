
import math
import numpy as np
import dezero
from dezero import optimizers
import dezero.functions as F
from dezero.models import MLP

# 1. 设置超参数
max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 1.0

# 2. 读入数据、创建模型和Optimizer
x, t = dezero.datasets.get_spiral(train=True)
model = MLP((hidden_size, 3))
optimizer = optimizers.SGD(lr).setup(model)

data_size = len(x)
max_iter = math.ceil(data_size / batch_size)

for epoch in range(max_epoch):
    # 3. 数据集索引重排
    index = np.random.permutation(data_size)
    sum_loss = 0

    for i in range(max_iter):
        # 4. 创建最小批量数据
        batch_index = index[i * batch_size:(i + 1) * batch_size]
        batch_x = x[batch_index]
        batch_t = t[batch_index]

        # 5. 算出梯度、更新参数
        y = model(batch_x)
        loss = F.softmax_cross_entropy(y, batch_t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(batch_t)

    # 6. 输出每轮的训练情况
    avg_loss = sum_loss / data_size
    print("epoch %d, loss %.2f" % (epoch + 1, avg_loss))






