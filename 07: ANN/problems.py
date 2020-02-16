import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit as sigmoid
import math
# import scipy
# import sklearn as sk
# import tensorflow as tf
# import keras

def universalApproximationTheorem (N, x, w, b, v):
    y = 0
    for i in range(N):
        y += v[i] * sigmoid(w[i]*x + b[i])
    return y

def createFunctions():
    N = 2
    x = np.linspace(0.0,1.0,10000)
    w = np.array([10000.0,10000.0])
    b = np.array([-4000., -6000.]) # shifting    
    v = np.array([0.8, -0.8]) # height
    y = universalApproximationTheorem(N, x, w, b, v)
    plt.plot(x,y)
    plt.show()

    N = 4
    x = np.linspace(0.0,1.0,10000)
    w = np.array([10000.0, 10000.0, 10000.0, 10000.0])
    b = np.array([-2000.0, -4000.0, -6000.0, -8000.0]) # shifting    
    v = np.array([0.5, -0.5, 0.2, -0.2]) # height
    y = universalApproximationTheorem(N, x, w, b, v)
    plt.plot(x,y)
    plt.show()

    N = 6
    x = np.linspace(0.0,1.0,10000)
    w = np.array([10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0])
    b = np.array([-2000.0, -4000.0, -4000.0, -6000.0, -6000.0, -8000.0]) # shifting    
    v = np.array([0.5, -0.5, 0.8, -0.8, 0.2, -0.2]) # height
    y = universalApproximationTheorem(N, x, w, b, v)
    plt.plot(x,y)
    plt.show()  
# createFunctions()

def recreateSine():
    N = 10000
    x = np.linspace(0.0,1.0,10000)
    y = 0
    for i in range(N):
        if(i<N/2):
            w = 10000.0
        else:   
            w = -10000.0
        b = x[int(abs(w)*(i+1)/N)-1]*w*-1.0
        v = math.sin(4*math.pi*abs((b/-w)))
        # print(v)
        y += v * sigmoid(w*x + b)
    plt.plot(x,y)
    plt.show()
recreateSine()