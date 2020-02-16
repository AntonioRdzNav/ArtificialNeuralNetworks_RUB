import numpy as np
import matplotlib.pyplot as plt
# import scipy
# import sklearn
# import tensorflow
# import keras

xs=[]
ys=[]
# y=0
# x=-50

# f(x) = ax + b
def quadratic(a, b, c):
    for x in range(-10,10,1):
        y=a*(x**2) + b*x + c
        xs.append(x)
        ys.append(y)
    plt.plot(xs,ys)
    plt.show()     

quadratic(5, 1, 0)
