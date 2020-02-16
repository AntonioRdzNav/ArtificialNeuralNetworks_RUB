import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
from matplotlib.cm import Paired
# import scipy
# import sklearn as sk
# import tensorflow as tf
# import keras

class MLP():
    
    def __init__(self, input_dim=3, hidden_layer_units=10, output_dim=1, eta=0.005):
    # input dimensions
        self.input_dim = input_dim
    # number of units in the hidden layer
        self.hidden_layer_units = hidden_layer_units
    # learning rate
        self.eta = eta
    # initialize weights randomly
        self.weights_input_hidden = np.random.normal(0, 1, (self.hidden_layer_units, input_dim))
        self.weights_hidden_output = np.random.normal(0, 1, (output_dim, self.hidden_layer_units))
    # use HE initialization for weights
        self.weights_hidden_output *= np.sqrt(2/input_size)
        self.weights_input_hidden *= np.sqrt(2/input_size)
    # initialize bias with vaue=1
        self.bias_input_hidden = np.zeros((hidden_layer_units, 1))
        self.bias_hidden_output = np.zeros((output_dim, 1))

    def ReLU(self, x):
        return np.maximum(0.0, x)
    def dReLU(self, x):
        x[x<=0] = 0
        x[x>0] = 1
        return x
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    def dsigmoid(self, x):
        return self.sigmoid(x) - (1-self.sigmoid(x))        

    def forwardPropagation(self, X):
    # from input to output (generates output)
        a_sum_input = np.dot(self.weights_input_hidden, X)+self.bias_input_hidden
        h_activ_input = self.sigmoid(a_sum_input)
        a_sum_hidden = np.dot(self.weights_hidden_output, h_activ_input)+self.bias_hidden_output
        prediction = self.sigmoid(a_sum_hidden)  
        return prediction, {
                    'a_sum_input': a_sum_input, 
                    'h_activ_input': h_activ_input, 
                    'a_sum_hidden': a_sum_hidden, 
                    'prediction': prediction
                  }

    def cost(self, predict, Y):
        m = Y.shape[1]
        cost__ = (1/m) * np.sum((Y-predict)**2)
        return (cost__)


    def backPropagation(self, X, Y, cache):
        m = X.shape[1]
    #   deltraTron = localGradient
        dpred = (cache['prediction'] - Y) * self.dsigmoid(cache['a_sum_hidden'])
        dweights_hidden_output = (1/m) * np.dot(dpred, np.transpose(cache['h_activ_input']))
        dbias_hidden_output = (1/m) * np.sum(dpred, axis=1, keepdims=True)
        da_sum_hidden = np.dot(np.transpose(self.weights_hidden_output), dpred) * self.dsigmoid(cache['a_sum_input'])
        dweights_input_hidden = (1/m) * np.dot(da_sum_hidden, np.transpose(X))
        dbias_input_hidden = (1/m) * np.sum(da_sum_hidden, axis=1, keepdims=True)
        return {
                "dweights_input_hidden": dweights_input_hidden, 
                "dbias_input_hidden": dbias_input_hidden, 
                "dweights_hidden_output": dweights_hidden_output, 
                "dbias_hidden_output": dbias_hidden_output
                }
            
    def updateParameters(self, gradients):
        self.weights_input_hidden   = self.weights_input_hidden - self.eta * gradients['dweights_input_hidden']
        self.weights_hidden_output  = self.weights_hidden_output - self.eta * gradients['dweights_hidden_output']        
        self.bias_input_hidden      = self.bias_input_hidden - self.eta * gradients['dbias_input_hidden']
        self.bias_hidden_output     = self.bias_hidden_output - self.eta * gradients['dbias_hidden_output']

    def train(self, X, Y, epochs=100):
        cost_ = []
        for j in range(epochs):
            prediction, cache = self.forwardPropagation(X)
            costit = self.cost(prediction, Y)
            gradients = self.backPropagation(X, Y, cache)
            self.updateParameters(gradients)
            cost_.append(costit)
            print("EPOCH: " + str(j) + " CALCULATED,\t COST = " + str(costit))
        return cost_, {
                "weights_input_hidden": self.weights_input_hidden, 
                "weights_hidden_output": self.weights_hidden_output, 
                "bias_input_hidden": self.bias_input_hidden, 
                "bias_hidden_output": self.bias_hidden_output
                }

    def MSE(self, y, prediction):
        summation = 0  #variable to store the summation of differences
        n = len(y) #finding total number of items in list
        for i in range (0,n):  #looping through each element of the list
            difference = y[i] - prediction[0][i]  #finding the difference between observed and predicted value
            squared_difference = difference**2  #taking square of the differene 
            summation =+ squared_difference  #taking a sum of all the differences
        MSE = summation/n  #dividing summation by total values to obtain average
        print ("The Mean Square Error is: " , MSE)

    def printDecisionBoundary(self, X, Y):
        prediction, cache = self.forwardPropagation(X.T)
        print(prediction.min(), prediction.max())

        prediction = (prediction*2)-1
        self.MSE(Y, prediction)

        prediction[prediction<=0] = -1
        prediction[prediction>0] = 1
        print(len(prediction[prediction==-1]), len(Y[Y==-1]))
        plt.scatter(X[:, 0], X[:, 1], c=prediction, alpha=0.3, cmap="jet")
        plt.show()


if __name__ == "__main__":
# load dataset
    dataset = np.load('xor.npy')
# prepare training data and labels
    X, Y = dataset[:,:2], dataset[:,2]
    # Y = np.where(Y==-1,0,1) #convert class labels to 0 & 1    
# add bias neuron (used as parameter instead)
    # bias = np.ones((X.shape[0],1))
    # X = np.concatenate((bias, X), axis=1)
# split into training and test
    X_train, X_test, Y_train, Y_test = X[:80000], X[80000:], Y[:80000], Y[80000:]
    X_train_T, Y_reshaped = X_train.T, Y_train.reshape(1, Y_train.shape[0])

    # plt.scatter(X_train_T.T[:,0], X_train_T.T[:,1], c=Y_train, alpha=0.3, cmap="jet")
    # plt.show()
    # plt.scatter(X_test[:,0], X_test[:,1], c=Y_test, alpha=0.3, cmap="jet")
    # plt.show()

    input_size = X_train_T.shape[0] # number of neurons in input layer      (=2)
    output_size = Y_reshaped.shape[0] # number of neurons in output layer    (=1)
    mlp = MLP(input_size, 16, output_size, 0.06)
    cost_, params = mlp.train(X_train_T, Y_reshaped, 100)
    plt.plot(cost_)
    plt.show()

    mlp.printDecisionBoundary(X_test, Y_test)

    # prediction, cache = mlp.forwardPropagation(X_test.T)
    # print(np.unique(prediction).size)