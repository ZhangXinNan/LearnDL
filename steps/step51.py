
import math
import numpy as np
import dezero
from dezero import optimizers
import dezero.functions as F
from dezero.dataloaders import DataLoader
from dezero.models import MLP


# 1. 设置超参数
max_epoch = 5
batch_size = 10
hidden_size = 1000
lr = 0.01

# 2. 读入数据、创建模型和Optimizer
train_set = dezero.datasets.MNIST(train=True)
test_set = dezero.datasets.MNIST(train=False)
train_loader = DataLoader(train_set, batch_size)
test_loader = DataLoader(test_set, batch_size, shuffle=False)

# model = MLP((hidden_size, 10))
model = MLP((hidden_size, hidden_size, 10), activation=F.relu)
optimizer = optimizers.SGD(lr).setup(model)

data_size = len(train_set)
max_iter = math.ceil(data_size / batch_size)
print(f"data_size: {data_size}, max_iter: {max_iter}")
for epoch in range(max_epoch):
    sum_loss, sum_acc = 0, 0

    for x, t in train_loader:
        # 5. 算出梯度、更新参数
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        acc = F.accuracy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(t)
        sum_acc += float(acc.data) * len(t)

    # 6. 输出每轮的训练情况
    avg_loss = sum_loss / len(train_set)
    avg_acc = sum_acc / len(train_set)
    # print(f"epoch: {epoch + 1}, train loss: {avg_loss:.4f}, accuracy: {avg_acc:.4f}")

    sum_loss, sum_acc = 0, 0
    with dezero.no_grad():
        for x, t in test_loader:
            # 5. 算出梯度、更新参数
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            acc = F.accuracy(y, t)
            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc.data) * len(t)
    test_avg_loss = sum_loss / len(test_set)
    test_avg_acc = sum_acc / len(test_set)
    print(f"epoch: {epoch + 1}, train loss: {avg_loss:.4f}, accuracy: {avg_acc:.4f}, test loss: {test_avg_loss:.4f}, accuracy: {test_avg_acc:.4f}")




