
import numpy as np
from dezero import Variable
import dezero.functions as F


x = Variable(np.array([1, 2, 3, 4, 5, 6]))
y = F.sum(x)
y.backward()
print(y, y.grad)
print(x, x.grad)


