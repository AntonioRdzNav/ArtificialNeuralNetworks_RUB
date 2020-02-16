# basic imports
import numpy as np
import matplotlib.pyplot as plt
# keras imports
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.datasets import fashion_mnist
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras import losses
from keras import backend as K


def buildModel(learning_rate=1, clipvalue=None):
    '''
    This function builds convolutional neural network model.
    
    | **Args**
    | learning_rate:      Learning rate to be used by the optimizer.
    '''
    # build model
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(3,3), input_shape=(28,28,1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(8, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    # compile model
    if clipvalue == None:
        model.compile(optimizer=SGD(lr=learning_rate), loss=losses.categorical_crossentropy, metrics=['accuracy'])
    else:
        model.compile(optimizer=SGD(lr=learning_rate, clipvalue=clipvalue), loss=losses.categorical_crossentropy, metrics=['accuracy'])
    
    return model

def reinitializeWeights(model=None):
    '''
    This function reinitializes the weights of a neural network model.
    For those who are interested in how to reinitialize weights without having to rebuild the whole model.
    May not work with recurrent models (cells of recurrent layers have additional recurrent kernels).
    
    | **Args**
    | model:              The model.
    '''
    session = K.get_session()
    # iterate over all layers
    for layer in model.layers:
        # if the layer has weights, run the initializer
        if hasattr(layer, 'kernel_initializer'): 
            layer.kernel.initializer.run(session=session)
        # if the layer has bias weights, run the initializer
        if hasattr(layer, 'bias_initializer'):
            layer.bias.initializer.run(session=session) 
            

if __name__ == "__main__":
    # load data set
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    # normalize examples
    X_train = X_train/255.
    X_test = X_test/255.
    # reshape training and test sets
    X_train = np.reshape(X_train, (X_train.shape[0], 28, 28, 1))
    X_test = np.reshape(X_test, (X_test.shape[0], 28, 28, 1))
    # transform labels
    y_train, y_test = to_categorical(y_train), to_categorical(y_test)
    # reduce data sets
    X_train, y_train = X_train[:20000], y_train[:20000]
    # number of epochs to be trained
    epochs=10
    # store results
    results = {1: [], 0.1: [], 0.01:[], 0.001:[], 0.0001: []}
    # train models with varying learning rates
    for learning_rate in [10**-i for i in range(5)]:
        model = buildModel(learning_rate=learning_rate)
        for epoch in range(epochs):
            model.fit(X_train, y_train, epochs=1, verbose=1)
            results[learning_rate] += [model.evaluate(X_train, y_train, verbose=0)]
    # convert lists to arrays
    for learning_rate in results.keys():
        results[learning_rate] = np.array(results[learning_rate])
    # plot results
    plt.figure(1)
    plt.title('Learning Rate Effects')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    for learning_rate in results.keys():
        plt.plot(results[learning_rate][:,0], label=('$\eta = %f$'%learning_rate))
    plt.xticks(np.arange(epochs), tuple([i+1 for i in range(epochs)]))
    axes = plt.gca()
    art = []
    lgd = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.15), ncol=3)
    art.append(lgd)
    plt.savefig('plots/learning_rate_loss.png', dpi=300, bbox_inches='tight')
    plt.figure(2)
    plt.title('Learning Rate Effects')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    for learning_rate in results.keys():
        plt.plot(results[learning_rate][:,1], label=('$\eta = %f$'%learning_rate))
    plt.xticks(np.arange(epochs), tuple([i+1 for i in range(epochs)]))
    axes = plt.gca()
    art = []
    lgd = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.15), ncol=3)
    art.append(lgd)
    plt.savefig('plots/learning_rate_accuracy.png', dpi=300, bbox_inches='tight')
    
    K.clear_session()