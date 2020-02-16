import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
from scipy.special import expit as sigmoid
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# import scipy
# import sklearn as sk
# import tensorflow as tf
# import keras

def logisticRegression():
    data = np.load("05_log_regression_data.npy")    # [feature1, feature2, label]
                                                        # feature1 = length of currency residency
                                                        # feature2 = yearly income
                                                        # label = whether a bank loan was granted or not
# standardization: is the process of putting different variables on the same scale
    feature1 = zscore(data[:,0])
    feature2 = zscore(data[:,1])
# split data
    x = np.column_stack((np.full(data.shape[0],1),feature1,feature2))# [bias, x1, x2]      
    y = data[:,2]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, train_size=0.8, random_state=10)
# plot data when y=0 and when y=1
    train_plot = np.column_stack((x_train,y_train))
    train_plot_0 = train_plot[train_plot[:,3] == 0] #train_plot column 3 == y_train
    train_plot_1 = train_plot[train_plot[:,3] == 1]
    plt.plot(train_plot_0[:,1], train_plot_0[:,2], 'ro')
    plt.plot(train_plot_1[:,1], train_plot_1[:,2], 'go')
# create model
    epoch = 15000
    model = LogisticRegression(max_iter=epoch)  # doesnt need LR because doesnt fit by SGD
# train model
    model.fit(x_train, y_train)
# make prediction and get MSE (just to check how good is the model)
    y_predTest = model.predict(x_test)
    MSE = model.score(x_test, y_test)
    print("MSE = ", str(MSE))
# get parameters
    parameters = model.coef_.reshape(3) # turn them from matrix (1row, 3 columns) to a vector (3 elements)
    print (parameters)
# plot decision boundary over original data
# knowing that o0 + o1*x1 + o2*p2 = 0 = b0 + b1*x + b2*y

# (x) values are just the min and max 2 points of the data (so that decisionBoundary is plottet through all data points) 
    x_values = [np.min(x[:, 1]), np.max(x[:, 2])] 
# after solving for (y) in (o0 + o1*x1 + o2*p2) = (0) = (b0 + b1*x + b2*y)
# (y) values are just 2 values of (y) calculated using the calculated parameters and the chosen (x) values
    y_values = - (parameters[0] + np.dot(parameters[1], x_values)) / parameters[2] 

    plt.plot(x_values, y_values,'bo') # only 2 points are plottet
    plt.plot(x_values, y_values, label='Decision Boundary') # lineal plot that intersects the 2 points previously mentioned
    plt.xlabel('Marks in 1st Exam')
    plt.ylabel('Marks in 2nd Exam')
    plt.legend()
    plt.show()
logisticRegression()


# def logisticRegression():
#     data = np.load("05_log_regression_data.npy")    # [feature1, feature2, label]
#                                                         # feature1 = length of currency residency
#                                                         # feature2 = yearly income
#                                                         # label = whether a bank loan was granted or not
# # standardization: is the process of putting different variables on the same scale
#     feature1 = zscore(data[:,0])
#     feature2 = zscore(data[:,1])
# # split data
#     x = np.column_stack((feature1,feature2))
#     y = data[:,2]
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, train_size=0.8, random_state=10)
# # plot data when y=0 and when y=1
#     train_plot = np.column_stack((x_train,y_train))
#     train_plot_0 = train_plot[train_plot[:,2] == 0]
#     train_plot_1 = train_plot[train_plot[:,2] == 1]
#     plt.plot(train_plot_0[:,0], train_plot_0[:,1], 'ro')
#     plt.plot(train_plot_1[:,0], train_plot_1[:,1], 'go')
# # minimize loss function using gradient decent
#     LR = 0.001
#     N = train_plot.shape[0]
#     parameters = [0, 0, 0]  # θbias + θx1 + θx2
#     x_gradient = np.column_stack((np.full(x_train.shape[0],1), x_train)) # [bias, x1, x2]    
#     epoch = 15000
#     # epoch = 1
#     for e in range(epoch):
#         sum = 0
#         for i in range(N):
#             sum += (sigmoid(np.matmul(np.transpose(parameters),x_gradient[i]))-y_train[i]) * x_gradient[i]
#         cost = (1/N) * sum
#         parameters = parameters - LR*(cost)
#         if(e%1000 == 0):
#             print(parameters)
#     # plt.show()
# logisticRegression()