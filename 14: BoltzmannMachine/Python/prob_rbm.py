# imports
import numpy as np

class RBM():
    '''
    This class implements a restricted Boltzmann machine (RBM).
    
    | **Args**
    | visible_units:         Number of visible units.
    | hidden_units:          Number of hidden units.
    | eta:                   The learning rate.
    '''
    
    def __init__(self, visible_units=64, hidden_units=16, eta=0.01):
        # unit counts
        self.visible_units = visible_units
        self.hidden_units = hidden_units
        # weights
        self.weights = np.random.normal(0, 0.01, (self.hidden_units, self.visible_units))
        self.visible_bias = np.random.normal(0, 0.01, (self.visible_units, 1))
        self.hidden_bias = np.random.normal(0, 0.01, (self.hidden_units, 1))
        # activation function
        self.activation_function = lambda x: 1. / (1. + np.exp(-x))
        # learning rate
        self.eta = eta
    
    def get_weights(self):
        '''
        Returns weights and biases.
        '''
        
        return {'weights': self.weights, 'visible_bias': self.visible_bias, 'hidden_bias': self.hidden_bias}
    
    def load_weights(self, weights):
        '''
        Loads weights and biases.
        
        | **Args**
        | weights:            Dictionary containing weights and biases for the RBM.
        '''
        
        assert self.weights.shape == weights['weights'].shape and self.visible_bias.shape == weights['visible_bias'].shape and self.hidden_bias.shape == weights['hidden_bias'].shape, 'weight matrix dimensions do not match!'
        
        self.weights = weights['weights']
        self.visible_bias = weights['visible_bias']
        self.hidden_bias = weights['hidden_bias']
    
    # Implement This Function
    def train(self, X, epochs=1):
        '''
        Train function of the RBM class.
        This functions trains a RBM using single-step contrastive divergence.
        
        | **Args**
        | X:                  Training examples.
        | epochs:             Number of epochs the MLP will be trained.
        '''
        pass