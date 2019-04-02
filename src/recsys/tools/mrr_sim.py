import numpy as np


score = np.random.random(10)

relevant = [0,0,0,0,1,0,0,0,0,0,0,0]

n = len(relevant)

for i in range(n):
    for j in range(n):
        print(i,j)