

if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


import numpy as np
from dezero import Variable
from dezero import sub, mul, add, pow


def sphere(x, y):
    z = x ** 2 + y ** 2
    return z


def matyas(x, y):
    # return 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y
    z = sub(mul(0.26, add(pow(x, 2), pow(y, 2))), mul(0.48, mul(x, y)))
    return z


def goldstein(x, y):
    z = (1 + (x + y + 1)**2 * (19 - 14*x + 3 * x**2 - 14*y + 6 * x * y + 3 * y **2))
    z = z * (30 + (2 * x - 3 * y)**2 * (18 - 32 * x + 12 * x**2 + 48 * y - 36 * x * y + 27 * y**2))
    return z


x = Variable(np.array(1.0))
y = Variable(np.array(1.0))
z = sphere(x, y)
z.backward()
print(z.data, z.grad)
print(x.data, y.data)
print(x.grad, y.grad)


x = Variable(np.array(1.0))
y = Variable(np.array(1.0))
z = matyas(x, y)
z.backward()
print(x.grad, y.grad)


x = Variable(np.array(1.0))
y = Variable(np.array(1.0))
z = goldstein(x, y)
z.backward()
print(x.grad, y.grad)



'''
# 计算图的定义
a = Variable('a')
b = Variable('b')
c = a * b
d = c + Constant(1)
# 计算图的编译
f = complie(d)
# 数据流转
d = f(a=np.array(2), b=np.array(3))
'''





