import numpy as np
import matplotlib.pyplot as plt
import keras
# import sklearn as sk

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.datasets import fashion_mnist

def fullyConnectedNetworks():
# a)
    # load data
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    # vectorize data
    origImgTrain = x_train
    origImgTest = x_test
    x_train.resize(x_train.shape[0], x_train.shape[1]**2)
    x_test.resize(x_test.shape[0], x_test.shape[1]**2)    

# b)
# pixels are given in range [0,255], must be normilized to [0,1]
# normalization is made to aid convergence (lower errors)
    x_train = x_train/255.0
    x_test = x_test/255.0

# c)
    model = keras.models.Sequential()
    model.add(Dense(units=10, activation='relu', input_dim=x_train.shape[1]))       # hidden-layer
    model.add(Dense(units=10, activation='relu', input_dim=x_train.shape[1]))       # hidden-layer
    model.add(Dense(units=10, activation='softmax', input_dim=x_train.shape[1]))    # output-layer

# d) 
    # (784 × 10 + 10) + (10 × 10 + 10) + (10 × 10 + 10) = 8070 
    # there are 8070 parameters in the whole network

# e)
    sgd = keras.optimizers.SGD(lr=0.005) # change the learning rate to 0.005
    model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# f)
    batch_size = 64
    epochs = 3
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test,y_test))
    # after 3 epochs:
        # train_loss: 0.5682 - train_acc: 0.8028 - val_loss: 0.5698 - val_acc: 0.8018

# g)
    print("Learning Rate: " + str(keras.backend.eval(model.optimizer.lr)))
    # LR was 0.01 but was changed to 0.005

# i)
    # reduce size of dataset
    x_train=x_train[:20000]
    y_train=y_train[:20000]
    # using this shorter dataset with more complex network causes OVERFITTING:
        # val_acc ">>" train_acc
    # it happens because the network is large enough that it can memorize the training examples
    # This is problematic, because we want to predict items that are not in the training dataset.
# h)
    newModel = keras.models.Sequential()
    newModel.add(Dense(units=64, activation='relu', input_dim=x_train.shape[1]))
    newModel.add(Dense(units=64, activation='relu', input_dim=x_train.shape[1]))
    newModel.add(Dense(units=10, activation='softmax', input_dim=x_train.shape[1]))
    # (784 × 64 + 64) + (64 × 64 + 64) + (64 × 10 + 10)) = 55050
        # the number of parameters increased to 55050    
    sgd = keras.optimizers.SGD(lr=0.01)
    newModel.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    newModel.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test,y_test))
    print("Learning Rate: " + str(keras.backend.eval(model.optimizer.lr)))
    # after 3 epochs:
        # train_loss: 0.6569 - train_acc: 0.7753 - val_loss: 0.6480 - val_acc: 0.7734  

# j) doesnt work because of problem in image dimensionality

    # predictions = np.argmax(newModel.predict(x_test), axis=1)
    
    # for i in range(predictions.shape[0]):
    #     if predictions[i] == 9:
    #         plt.imshow(origImgTest[i])
    #         plt.pause(0.01)
    #         if y_test[i]!=9:
    #             print("WRONG classification! True class of this item is %d." % y_test[i])
    #             plt.show()
    #         else:
    #             print("Item correctly classified!")
    #         plt.cla()

# fullyConnectedNetworks()

def CNN():
# a)
    # load data
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    # pixels are given in range [0,255], must be normilized to [0,1]
    # normalization is made to aid convergence (lower errors)
    x_train = x_train/255.0
    x_test = x_test/255.0    
    # make data 28x28x1 so that it can serve as CNN input
    x_train = x_train.reshape(x_train.shape[0],28,28, 1)
    x_test = x_test.reshape(x_test.shape[0],28,28, 1)
    input_shape = (x_train[0].shape[0], x_train[0].shape[1], 1)
    # create CNN
    CNN = keras.models.Sequential([
        Conv2D(filters=16, kernel_size=(3,3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2,2)),
        Conv2D(filters=8, kernel_size=(3,3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2,2)),
        Flatten(),  #Flatten serves as a connection between the convolution and dense layers
        Dense(units=10, activation='softmax') # Dense is a standard layer type 
    ])    

# b)
    # sgd = keras.optimizers.SGD(lr=0.005) # change the learning rate to 0.005
    CNN.compile(loss='sparse_categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])

    batch_size = 64
    epochs = 3
    CNN.fit(x_train, y_train, epochs=epochs, validation_data=(x_test,y_test), batch_size=batch_size)
    print("Learning Rate: " + str(keras.backend.eval(CNN.optimizer.lr)))
    # after 3 epochs:
        # train_loss: 0.5050 - train_acc: 0.8163 - val_loss: 0.5089 - val_acc: 0.8160

# CNN()


def bonus_CNN():
# a)
    # load data
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    # pixels are given in range [0,255], must be normilized to [0,1]
    # normalization is made to aid convergence (lower errors)
    x_train = x_train/255.0
    x_test = x_test/255.0    
    # make data 28x28x1 so that it can serve as CNN input
    x_train = x_train.reshape(x_train.shape[0],28,28, 1)
    x_test = x_test.reshape(x_test.shape[0],28,28, 1)
    input_shape = (x_train[0].shape[0], x_train[0].shape[1], 1)
    # create CNN
    CNN = keras.models.Sequential([
        Conv2D(filters=16, kernel_size=(3,3), strides=(2, 2), activation='relu', input_shape=input_shape),
        Conv2D(filters=8, kernel_size=(3,3), strides=(2, 2), activation='relu', input_shape=input_shape),
        Flatten(),  #Flatten serves as a connection between the convolution and dense layers
        Dense(units=10, activation='softmax') # Dense is a standard layer type 
    ])    
    # sgd = keras.optimizers.SGD(lr=0.005) # change the learning rate to 0.005
    CNN.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    batch_size = 64
    epochs = 3
    CNN.fit(x_train, y_train, epochs=epochs, validation_data=(x_test,y_test), batch_size=batch_size)
    print("Learning Rate: " + str(keras.backend.eval(CNN.optimizer.lr)))
    # after 3 epochs:
        # train_loss: 0.4029 - train_acc: 0.8557 - val_loss: 0.4181 - val_acc: 0.8489
bonus_CNN()
