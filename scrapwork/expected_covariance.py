import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt

rng = default_rng(1001)
size = 10
repeats = 10**6

mat = np.zeros([size, size])

for _ in range(repeats):
    vec = rng.standard_normal(size)
    mat += np.outer(vec, vec)

plt.matshow(mat/repeats)
plt.show()
