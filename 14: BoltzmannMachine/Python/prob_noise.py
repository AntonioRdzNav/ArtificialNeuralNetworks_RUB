# imports
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from prob_rbm import RBM


if __name__ == "__main__":
    # test for different noise levels
    noise_levels = [0., 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    
    # load sklearn's digit data set
    digits  = load_digits()
    
    # prepare data
    X = digits.data/16
    X_test = X[1200:]
    X = X[:1200]
    
    # implement !!!