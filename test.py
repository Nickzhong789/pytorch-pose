import torch
import numpy as np
from fractions import Fraction as fr

a = np.matrix([
    [1, -4, -3],
    [0, 1, 1],
    [-1, 6, 4]
])
b = np.matrix([
    [4, 2, 3],
    [1, 1, 0],
    [-1, 2, 3]
])
# print(a)
# print(a.I)
print(np.dot(a, b))
