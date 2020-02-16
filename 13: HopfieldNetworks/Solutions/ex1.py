import numpy as np
from general import sign
        
w = np.array([[0, -1, -1, 1],
              [-1, 0, 1, -1],
              [-1, 1, 0, -1],
              [1, -1, -1, 0]])
    
b = 0
    
xprime = np.array([1, -1, 1, -1])   # Initial guess (random pattern)

num_iter = 10                       # Number of iterations
energy = np.zeros(num_iter)         # Initializing energy vector
J = np.zeros(num_iter)              # Initializing a vector to store random indicies for asynchronous update.

for n_iter in range(num_iter):
    j = np.random.randint(0, xprime.size)
    J[n_iter] = j
    xprime[j] = sign(w[j, :].dot(xprime) + b)
    print(xprime)
    energy[n_iter] = -0.5*np.dot(xprime.dot(w), xprime.T) - b

print("The energy vector is: ", energy)
print("The random indicies in each oteration", J)


# a) The stored pattern had 4 elements. Because weight matrix has 4 rows (and 4 columns)