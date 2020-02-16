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
        self.weights_input_hidden = np.random.normal(0, 1, (input_dim, self.hidden_layer_units))
        self.weights_hidden_output = np.random.normal(0, 1, (self.hidden_layer_units, output_dim))
    # use HE initialization for weights
        # self.weights_hidden_output *= np.sqrt(2/input_size)
        # self.weights_input_hidden *= np.sqrt(2/input_size)
    # initialize bias with vaue=1
        self.bias_input_hidden = np.zeros((hidden_layer_units))
        self.bias_hidden_output = np.zeros((output_dim))

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
        a_sum_input = np.dot(X, self.weights_input_hidden)+self.bias_input_hidden
        h_activ_input = self.sigmoid(a_sum_input)
        a_sum_hidden = np.dot(h_activ_input, self.weights_hidden_output)+self.bias_hidden_output
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
    #   deltraTron = localGradient
        dpred = np.multiply(-(Y-cache['prediction']), self.dsigmoid(cache['a_sum_hidden'])) #element-wise matrix multiplication
        dweights_hidden_output = np.dot(np.transpose(cache['h_activ_input']), dpred)
        dbias_hidden_output = np.sum(dpred, axis=0, keepdims=True)

        da_sum_hidden = np.dot(dpred, np.transpose(self.weights_hidden_output)) * self.dsigmoid(cache['a_sum_input'])
        dweights_input_hidden = np.dot(np.transpose(X), da_sum_hidden)
        dbias_input_hidden = np.sum(da_sum_hidden, axis=0, keepdims=True)
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
            if(j%1 == 0):
                print("EPOCH: " + str(j) + " CALCULATED,\t COST = " + str(costit))
        return cost_, {
                "weights_input_hidden": self.weights_input_hidden, 
                "weights_hidden_output": self.weights_hidden_output, 
                "bias_input_hidden": self.bias_input_hidden, 
                "bias_hidden_output": self.bias_hidden_output
                }

    def printDecisionBoundary(self, X, Y):
        # h = .1  # step size in the mesh
        # create a mesh to plot in
        # x_min, x_max = X[:, 0].min(), X[:, 0].max()
        # y_min, y_max = X[:, 1].min(), X[:, 1].max()
        # xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
        #                     np.arange(y_min, y_max, h))

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, m_max]x[y_min, y_max].
        # fig, ax = plt.subplots()

        prediction, cache = self.forwardPropagation(X)
        print(prediction.min(), prediction.max())

        # x = np.arange(0, Y.shape[0], 1)
        # error = np.abs(Y-prediction).reshape(Y.shape[0])
        # plt.plot(x, error)
        # plt.show()
        # print(error[error==0].shape)

        # Put the result into a color plot
        # prediction = prediction.reshape(xx.shape)
        # ax.contourf(xx, yy, prediction, cmap="jet")
        # ax.axis('off')

        # Plot also the training points
        prediction[prediction==0] = -1
        prediction[prediction>0] = 1
        print(len(prediction[prediction==-1]), len(Y[Y==-1]))
        plt.scatter(X[:, 0], X[:, 1], c=prediction, alpha=0.3, cmap="jet")

        # ax.set_title('Perceptron')
        plt.show()



if __name__ == "__main__":
# load dataset

# prepare training data and labels

    # Y = np.where(Y==-1,0,1) #convert class labels to 0 & 1    
# add bias neuron (used as parameter instead)
    # bias = np.ones((X.shape[0],1))
    # X = np.concatenate((bias, X), axis=1)
# split into training and test


    dataset = np.load('xor.npy')
    X, Y = dataset[:,:2], dataset[:,2]
    X_train, X_test, Y_train, Y_test = X[:80000], X[80000:], Y[:80000], Y[80000:]
    Y_train = Y_train.reshape(Y_train.shape[0], 1)
    Y_test = Y_test.reshape(Y_test.shape[0], 1)
    input_size = X_train.shape[1] # number of neurons in input layer      (=2)
    output_size = Y_train.shape[1] # number of neurons in output layer    (=1)
    mlp = MLP(input_size, 3, output_size, 0.05)
    cost_, params = mlp.train(X_train, Y_train, 1000)
    plt.plot(cost_)
    plt.show()    
    mlp.printDecisionBoundary(X, Y)


    # plt.scatter(X_train_T.T[:,0], X_train_T.T[:,1], c=Y_train, alpha=0.3, cmap="jet")
    # plt.show()
    # plt.scatter(X_test[:,0], X_test[:,1], c=Y_test, alpha=0.3, cmap="jet")
    # plt.show()

    # X = np.random.rand(10000,2)
    # Y = np.apply_along_axis(lambda element: element[0]+element[1], axis=1, arr=X).reshape(X.shape[0],1)

    # input_size = X.shape[1] # number of neurons in input layer      (=2)
    # output_size = Y.shape[1] # number of neurons in output layer    (=1)
    # mlp = MLP(input_size, 3, output_size, 0.01)
    # cost_, params = mlp.train(X, Y, 3000)
    # plt.plot(cost_)
    # plt.show()

    # mlp.printDecisionBoundary(X, Y)

    # prediction, cache = mlp.forwardPropagation(X_test.T)
    # print(np.unique(prediction).size)