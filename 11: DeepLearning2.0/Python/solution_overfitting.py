# basic imports
import numpy as np
import matplotlib.pyplot as plt
# keras imports
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.datasets import fashion_mnist
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras import losses
from keras import backend as K


def buildModel():
    '''
    This function builds convolutional neural network model.
    '''
    # build model
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(3,3), input_shape=(28,28,1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(8, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    # compile model
    model.compile(optimizer=Adam(lr=0.01), loss=losses.categorical_crossentropy, metrics=['accuracy'])
    
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
    # number of epochs to be trained
    epochs = 10
    # number of runs
    runs = 1
    # store results
    results_train = {30000: [], 15000: [], 2500: [], 500: []}
    results_test = {30000: [], 15000: [], 2500: [], 500: []}
    # train models
    for size in results_train.keys():
        print('Training Samples: %d' % size)
        for run in range(runs):
            print('\tRun: %d' % (run+1))
            model = buildModel()
            model.fit(X_train[:size], y_train[:size], epochs=epochs, verbose=0)
            results_train[size] += [model.evaluate(X_train[:size], y_train[:size], verbose=0)]
            results_test[size] += [model.evaluate(X_test, y_test, verbose=0)]
    # evaluate
    for size in results_train.keys():
        results_train[size] = np.mean(np.array(results_train[size]), axis=0)
        results_test[size] = np.mean(np.array(results_test[size]), axis=0)
    # prepare results for plotting
    accuracy_train, accuracy_test, loss_train, loss_test = [], [], [], []
    for size in results_train.keys():
        accuracy_train += [results_train[size][1]]
        accuracy_test += [results_test[size][1]]
        loss_train += [results_train[size][0]]
        loss_test += [results_test[size][0]]
    # plot results
    plt.figure(1)
    plt.title('Accuracy')
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy')
    plt.plot(accuracy_train, label='Training Set')
    plt.plot(accuracy_test, label='Test Set')
    plt.xticks(np.arange(4), ('30000','15000','2500','500'))
    axes = plt.gca()
    art = []
    lgd = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.15), ncol=3)
    art.append(lgd)
    plt.savefig('plots/overfitting_accuracy.png', dpi=300, bbox_inches='tight')
    plt.figure(2)
    plt.title('Loss')
    plt.xlabel('Training Set Size')
    plt.ylabel('Loss')
    plt.plot(loss_train, label='Training Set')
    plt.plot(loss_test, label='Test Set')
    plt.xticks(np.arange(4), ('30000','15000','2500','500'))
    axes = plt.gca()
    art = []
    lgd = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.15), ncol=3)
    art.append(lgd)
    plt.savefig('plots/overfitting_loss.png', dpi=300, bbox_inches='tight')
    
    K.clear_session()