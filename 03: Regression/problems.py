import numpy as np
import matplotlib.pyplot as plt
# import scipy
from sklearn.linear_model import LinearRegression
# import tensorflow as tf
# import keras


# Analytical solution for linear regression
# y = 4x + 5 + e
# a)
def linearRegression():
    N = 100
    x = np.linspace(-3, 3, num=N, dtype=float, endpoint=True)
    e = np.random.uniform(low=-0.5, high=0.5, size=N)
    y = (4*x + 5 + e)
    plt.subplot(1,2,1)
    plt.plot(x,y, 'ro')
# b)
    m = (np.sum(x*y) - (np.sum(x)*np.sum(y))/N) / (np.sum(x**2) - (np.sum(x)**2)/N)
    b = np.sum(y)/N - np.sum(x)*m/N
# c)
    x0 = np.full(100, 1)
    X = np.column_stack((x0, x))
    param = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(X), X)), np.transpose(X)), y)
# d)
    y_pred = np.matmul(X, param)
    plt.plot(x, y_pred, 'b')
    SS_data = np.sum((y - np.average(y))**2)
    SS_reg = np.sum((y_pred - np.average(y_pred))**2)
    R_2 = (SS_reg / SS_data) ** 2
    residual = y - y_pred
    plt.subplot(1,2,2)
    plt.plot(x, residual, 'ro')
    # print(R_2)
    plt.show()    
# e)
    linearRegressor = LinearRegression()
    x_train = x.reshape((-1, 1))
    y_train = y_pred
    linearRegressor.fit(x_train, y_train)
    N = 200
    x = np.linspace(-10, 10, num=N, dtype=float, endpoint=True)
    e = np.random.uniform(low=-6.0, high=6.0, size=N)
    y = (4*x + 5 + e)   
    x_train = x.reshape((-1, 1))
    y_pred_sk = linearRegressor.predict(x_train)
    plt.plot(x, y, 'ro')    
    plt.plot(x, y_pred_sk)   
    R_2 = linearRegressor.score(x_train, y_pred_sk)
    # print(R_2) 
    plt.show()    
# linearRegression()


# Polynomial Regression
# y = -x^5 + 1.5x^3 - 2x^2 + 4x + 3 + e
def polynomialRegression():
# a)
    N = 100
    x = np.linspace(-2, 2, num=N, dtype=float, endpoint=True)
    e = np.random.uniform(low=-1.0, high=1.0, size=N)
    y = (-1*x**5 + 1.5*x**3 - 2*x**2 + 4*x + 3 + e)
    # plt.subplot(1,2,1)
    plt.subplot(1,2,1)
    plt.plot(x,y, 'ro') 
# b)
    x0 = np.full(N, 1)
    X = np.column_stack((x0, x, x**2, x**3, x**4, x**5))
    param = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(X), X)), np.transpose(X)), y)
    y_pred = np.matmul(X, param)
    plt.plot(x, y_pred, 'b')
# c)
    plt.subplot(1,2,2)
    plt.plot(x, y-y_pred, 'ro')
    plt.show()
# polynomialRegression()   

# Incremental Version of Linear Regression
# y = 4x + 5 + e
def stochasticLinearRegression():
# a)
    LR = 0.001  
    epoch = 50 # (number of itterations)
    N = 100
    m_new = 0
    b_new = 0   
    x = np.linspace(-3, 3, num=N, dtype=float, endpoint=True) 
    e = np.random.uniform(low=-0.5, high=0.5, size=N)
    y = (4*x + 5 + e)
    for i in range(epoch):
        y_new = (m_new*x) + 5 + b_new 
        # cost = np.sum((y-y_new)**2)
        b_grad = -2 * np.sum((y-y_new))
        m_grad = -2 * np.sum(x*(y-y_new))
        b_new = b_new - (LR * b_grad)
        m_new = m_new - (LR * m_grad)
        y_withNewParameters = m_new*x + 5 + b_new 
        plt.plot(x,y, 'ro')
        plt.plot(x,y_withNewParameters, 'b')
        plt.pause(1)
# stochasticLinearRegression()
