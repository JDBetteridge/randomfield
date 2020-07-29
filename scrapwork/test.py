import numpy as np

from numpy.linalg import cholesky

A = np.array([[  4,  12, -16],
              [ 12,  37, -43],
              [-16, -43,  98]])
print(A)

H = cholesky(A)
print(H)
