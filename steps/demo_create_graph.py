

import numpy as np
from dezero import Variable
x = Variable(np.array(2.0))
y = x ** 4
# y.backward()
y.backward(create_graph=True)

print(x)
print(x.grad)  # variable(32.0)
print(y)
print(y.grad)


gx = x.grad
x.cleargrad()
gx.backward()
print(x.grad)


