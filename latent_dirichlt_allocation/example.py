__author__ = 'billhuang'

import numpy as np
import LDA

# SET RANDOM SEED
np.random.seed(1234)

# GENERATE DATA
# word to be 0 - 4

print('generate sample...')

N = [1000, 1000, 1000]

theta = np.array([[0.9, 0.05, 0.05],
                  [0.1, 0.7, 0.2],
                  [0.1, 0.2, 0.7]])

beta = np.array([[0, 0.3, 0, 0.6, 0.1],
                 [0.8, 0.05, 0.05, 0.05, 0.05],
                 [0.05, 0.05, 0.5, 0, 0.4]])

print('theta')
print(theta)
print('beta')
print(beta)
print()

Y = []

for i in range(3):
    yi = np.zeros(N[i], dtype = int)
    for j in range(N[i]):
        topic = np.random.choice(3, p = theta[i,:])
        yi[j] = np.random.choice(5, p = beta[topic,:])
    Y.append(yi)
    
LDA.LDA(Y, 3, 3, 5)
