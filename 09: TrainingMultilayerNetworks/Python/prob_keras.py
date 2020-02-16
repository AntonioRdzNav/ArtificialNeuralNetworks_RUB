from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras import backend
import numpy as np
#load and split dataset
dataset = np.load('xor.npy')
X, y = dataset[:,:2], dataset[:,2]
y = np.where(y==-1,0,1) #convert class labels to 0 & 1
bias = np.ones((X.shape[0],1))
X = np.concatenate((bias, X), axis=1)
y_enc= to_categorical(y)
X_train, X_test, y_train, y_test = X[:80000], X[80000:], y_enc[:80000], y_enc[80000:]

mse = []
for run in range(100) :
#create the network
    model = Sequential()
    # Dense layer type means that all neurons are connected to each other
    # second layer to create has 64 units (input layer contains 3 units)
    model.add(Dense(64,input_dim=3))
    model.add(Activation('tanh'))
    # third layer to create has 2 units
    model.add(Dense(2))
    model.add(Activation('softmax'))
    #add optimizer, compile and train
    sgd = SGD(lr=0.005)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['mse'])
    #evaluate loss and predict classes on test set
    model.fit(X_train, y_train, batch_size=1, epochs=1)
    loss = model.evaluate(X_test,y_test)
    mse.append(loss[1])
    backend.clear_session()
mse_mean = np.mean(mse)
mse_std = np.std(mse)