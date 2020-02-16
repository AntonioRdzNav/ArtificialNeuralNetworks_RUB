# imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from prob_rbm import RBM


if __name__ == "__main__":
    # load sklearn's digit data set
    digits  = load_digits()
    # prepare data
    X = digits.data/16
    X_test = X[1200:]
    X = X[:1200]
    
    # implement !!!