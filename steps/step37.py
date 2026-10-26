
import numpy as np
import dezero.functions as F
from dezero import Variable


# x = Variable(np.array(1.0))
x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
y = F.sin(x)
print(y)

c = Variable(np.array([[10, 20, 30], [40, 50, 60]]))
y = x + c
print(y)

# t = x + c
# y = F.sum(t)
# print(y)
