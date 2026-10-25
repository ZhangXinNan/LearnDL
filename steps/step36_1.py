
import numpy as np
from dezero import Variable
import dezero.functions as F


x = Variable(np.array([1.0, 2.0]))
v = Variable(np.array([3.0, 4.0]))

def f(x):
    t = x ** 2
    y = F.sum(t)
    return y


y = f(x)
y.backward(create_graph=True)

gx = x.grad
x.cleargrad()

z = F.matmul(v, gx)
z.backward()
print(x.grad)