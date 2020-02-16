import numpy as np
import matplotlib.pyplot as plt
import scipy as sc

from sklearn.datasets import make_circles
from IPython.display import clear_output

class neural_layer():
    # n_conn = neurons in previous layer
    # n_neur = neurons in current layer
    # act_f  = activation function to apply on current layer
    def __init__(self, n_conn, n_neur, act_f):
        self.act_f = act_f
        self.b = np.random.rand(1, n_neur) * 2 - 1      # shape = [1, n_neur]      # *2-1 changes range [0,1] to [-1,1]
        self.W = np.random.rand(n_conn, n_neur) * 2 - 1 # shape = [n_conn, n_neur] # *2-1 changes range [0,1] to [-1,1]


#######################
# Constants
#######################
n = 1000 # n of samples
p = 2   # n of inputs (colums of X)
#######################

#######################
# Useful operations
#######################
# Activation functions
sigm = (lambda x: 1 / (1+np.e ** (-x)),     #activation function sigmoid    #access by sigm[0](x_value)
        lambda x: x * (1-x))                #derivate of sigmoid            #access by sigm[1](x_value)

def relu_d(x):
    x[x>0] = 1
    x[x<=0] = 0
    return x
relu = (lambda x: np.maximum(0,x),          #activation function relu       #access by relu[0](x_value)
        lambda x: relu_d(x))      #derivate of relu               #access by relu[1](x_value)
# Cost function
l2_cost = (lambda pred, y:  np.mean((pred - y)**2),     #cost function          #access by l2_cost[0](pred, y)
           lambda pred, y: (pred - y))                  #derivate of cost_func  #access by l2_cost[1](pred, y)
########################





def create_nn(topology, act_functions):
    nn = [] # will represent a ANN with all the created layers inside
    for i, layer in enumerate(topology[:-1]): #[:-1] iterate until (last value -1)
        nn.append(neural_layer(topology[i], topology[i+1], act_functions[i])) #[i]=n_conn, [i+1]=n_neur
    return nn

def train(neural_net, X, Y, l2_cost, LR=0.5, train=True):
    # out = [(None,X), (z0,a0), (z1,a1), ... (zn,an)]
    out = [(None, X)]

# Forward Pass
    for i, layer in enumerate(neural_net):
        # out[-1] stands for last element of list
        # (@) == np.dot
        z = out[-1][1] @ neural_net[i].W + neural_net[i].b
        a = neural_net[i].act_f[0](z)
        out.append((z,a))

    if(train):  
# Backward Pass      (next = layerToTheRight), (previous = layerToTheLeft)
        deltas = []
        for i in reversed(range(0, len(neural_net))):
            z = out[i+1][0]     #[i+1] because we added (None,X) of input layer
            a = out[i+1][1]     #[i+1] because we added (None,X) of input layer

            if(i == len(neural_net)-1):
            # Calculate delta for last layer
                # lastDelta = derivateOfLostFunction(a,Y) * derivateOfActivationFunction(a)
                lastDelta = l2_cost[1](a,Y) * neural_net[i].act_f[1](a)
                deltas.insert(0, lastDelta) # insert because we want to place them at index 0 of list(front) (append place it at end)
            else:
            # Calculate delta for other layers
                # delta = previousDelta @ weightsOfNextLayer * derivateOfActivationFunction(a)              
                delta = deltas[0] @ _W.T * neural_net[i].act_f[1](a) #previousDelta = deltas[0] (always because of insert)
                deltas.insert(0, delta)  
            _W = neural_net[i].W  

# Gradient Descent
            # bias = bias - LR*ParcialDerivateOfCostFunction(bias)
            # bias = bias - LR*delta*1
            neural_net[i].b = neural_net[i].b - LR*np.mean(deltas[0], axis=0, keepdims=True)
            # weights = weights - LR*ParcialDerivateOfCostFunction(weights)
            # weights = weights - LR*delta*
            neural_net[i].W = neural_net[i].W - LR*(out[i][1].T @ deltas[0])        

    return out[-1][1]  #return prediction vector

def train_nEpochs_display(neural_net, X, Y, l2_cost, LR, epoch):
    loss = []
    for i in range(epoch):
        pred = train(neural_net, X, Y, l2_cost, LR, train=True)

        if(i%25 == 0):
            loss.append(l2_cost[0](pred, Y))

            resolution = 50
            _x0 = np.linspace(-1.5, 1.5, resolution)
            _x1 = np.linspace(-1.5, 1.5, resolution)
            _Y = np.zeros((resolution, resolution))

            for i0, x0 in enumerate(_x0):
                for i1, x1 in enumerate(_x1):
                    _Y[i0, i1] = train(neural_net, np.array([[x0,x1]]), Y, l2_cost, train=False)[0][0]
            
            plt.subplot(1,2,1)
            plt.pcolormesh(_x0, _x1, _Y, cmap="PiYG")
            plt.axis("equal")
            plt.scatter(X[Y[:,0]==0,0], X[Y[:,0]==0,1], color="red")
            plt.scatter(X[Y[:,0]==1,0], X[Y[:,0]==1,1], color="blue")
            plt.title("Classification")

            plt.subplot(1,2,2)
            plt.plot(range(len(loss)), loss)
            plt.title("Loss Vs Time")

            clear_output(wait=True)
            plt.pause(0.5)
        



def main():
# CREATE DATASET
    X, Y = make_circles(n_samples=n, factor=0.5, noise=0.05)
    Y = Y.reshape(n,1)
# PLOT DATASET
    plt.scatter(X[Y[:,0]==0,0], X[Y[:,0]==0,1], color="red")
    plt.scatter(X[Y[:,0]==1,0], X[Y[:,0]==1,1], color="blue")
    plt.axis("equal")
    plt.show()

# Create ANN
    #topology of ANN (lenght of topology is amount of layers the ANN has)
    #                (value of every cell is the amount of neurons that layer has)
    # topology = [p, 4, 8, 16, 8, 4, 1]   
    topology = [p, 5, 5, 1]   
    #act_functions of every layer (sigm or relu)
    # act_functions = [sigm,sigm,sigm,sigm,sigm,sigm ] 
    act_functions = [sigm,sigm,sigm]  #act_functions.lenght = topology.length-1
    neural_net = create_nn(topology, act_functions)

# Train ANN
    epoch = 2500
    train_nEpochs_display(neural_net, X, Y, l2_cost, 0.05, epoch) #LR=0.05 for sigm, LR=0.0006 for relu

if __name__ == "__main__":
    main()