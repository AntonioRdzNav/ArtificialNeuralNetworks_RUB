import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
# import scipy
# import sklearn as sk
# import tensorflow as tf
# import keras

def perceptronAlgorithm(X, y_real):
# init
    N = 3
    LR = 0.02
    W = np.random.rand(N) * 0.1 - 0.05
    # print("initW = " + str(W))
# add bias
    X = np.column_stack((np.ones(X.shape[0]), X))    
# calculate y_pred
    y_pred = np.matmul(np.reshape(W,(1,3)), X.T).reshape(X.shape[0])
    # y_pred = np.sign(y_pred)
    y_pred[y_pred <= 0] = -1
    y_pred[y_pred > 0] = 1
    learningW = [W]
    epoch = 5000
    targetError = 0.05
    errHist = []
    for n in range(epoch):
        # calculate prediction (y_pred)
        y_pred = np.matmul(np.reshape(W,(1,3)), X.T).reshape(X.shape[0])
        err = 0
        precision = 0
        for i in range(y_pred.shape[0]):
            if(y_pred[i]*y_real[i] < 0):
                W = W + 2*LR*y_real[i]*X[i]
                learningW.append(W)
                err += 1
                precision = err/X.shape[0]
                errHist.append(precision)
        if (1-precision) <= targetError:
            break  
    print("finalW = " + str(W))
    W = learningW[np.argmin(errHist)-1] # take best W found
    return W, errHist 
    

def perceptron():
# a)
# GET DATA
    datasets = load_iris()
    data = datasets.data[:, 0:2]
    targets = datasets.target
    targets[targets == 0] = -1    
    targets[targets == 2] = 1    
    # print(data.shape)
    # print(targets.shape)    #-1='setosa', 1='versicolor'|'virginica'
# SEPARATE DATA
    setosa = data[targets == -1]
    notSetosa = data[targets == 1]
    # print(setosa.shape)
    # print(notSetosa.shape)
# PLOT DATA
    plt.subplot(1,2,1)
    plt.plot(setosa[:,0], setosa[:,1], 'ro')
    plt.plot(notSetosa[:,0], notSetosa[:,1], 'bo')
    # plt.show()

# b)
    W, errHist = perceptronAlgorithm(data, targets)
    # for i in range(learningW.shape[0]):
    #     if i%150 == 0:
    #         W_ = learningW[i]
    x0 = W[0]
    x1 = W[1]
    x2 = W[2]
    x2_intercept = -x0/x2
    x1_intercept = -x0/x1
    m = -x2_intercept/x1_intercept
    b = x2_intercept
    x_values = [np.min(data[:, 0]), np.max(data[:, 0])] 
    y_values = np.dot(m,x_values) + b
    plt.plot(x_values, y_values,'go') # only 2 points are plottet
    plt.plot(x_values, y_values, label='Decision Boundary') # lineal plot that intersects the 2 points previously mentioned
    plt.xlabel('Marks in 1st Exam')
    plt.ylabel('Marks in 2nd Exam')
    plt.legend()  
    plt.subplot(1,2,2)
    plt.plot(range(len(errHist)),errHist)
    plt.show()        
perceptron()