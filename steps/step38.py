
import numpy as np
from dezero import Variable
import dezero.functions as F


x = Variable(np.array([[1,2,3], [4,5,6]]))
y = F.reshape(x, (6,))
y.backward(retain_grad=True)
print(x)
print(x.grad)
print(y)
print(y.grad)


x = Variable(np.random.randn(1, 2, 3))
print(x)

y = x.reshape((2, 3))
print(y)

y = x.reshape(2, 3)
print(y)


print("---transpose---")
x = Variable(np.array([[1,2,3], [4,5,6]]))
y = F.transpose(x)
y.backward()
print(x)
print(x.grad)
print(y)
print(y.grad)

print("===transpose===")
# print(x)
y = x.transpose()
print(y)
y = x.T
print(y)