# basic imports
import numpy as np
import pickle
# keras imports
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.datasets import fashion_mnist
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
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
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    # compile model
    model.compile(optimizer=Adam(lr=0.01), loss=losses.categorical_crossentropy, metrics=['accuracy'])
    
    return model
        

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
    # make a reduced data set
    X_reduced, y_reduced = np.copy(X_train[:10000]), np.copy(y_train[:10000])
    # number of epochs to be trained
    epochs = 1
    # training samples
    N = 60000
    # prepare data generator
    data_generator = ImageDataGenerator(
        rotation_range=20, 
        width_shift_range=0.1, 
        height_shift_range=0.1, 
        shear_range=0.5, 
        zoom_range=(0.9,1.1), 
        horizontal_flip=False, 
        vertical_flip=False, 
        fill_mode='constant',
        cval=0
    )
    data_generator = data_generator.flow(X_reduced, y_reduced)
    
    # a)
    # generate data 
    counter = 0
    Xs, ys = [], []
    for i, j in data_generator:
        # retrieve augmented batch and append to list
        Xs.append(i)
        ys.append(j)
        # increment counter
        counter += 1
        # stop
        if counter == 666:
            break
    # concatenate all the batches into a numpy array
    X_pret = np.concatenate(Xs)
    y_pret = np.concatenate(ys)
    # get rid of lists
    del Xs
    del ys
    # save augmented data set
    pickle.dump({'X': X_pret, 'y': y_pret}, open('fashion_mnist_augmented.pkl', 'wb'))
    # split augmented data set
    X_train_pret, y_train_pret, X_test_pret, y_test_pret = X_pret[:10000], y_pret[:10000], X_pret[10000:], y_pret[10000:]
    
    # b)
    # train model on normal training set
    model_orig = buildModel()
    model_orig.fit(X_train, y_train, epochs=epochs, verbose=0)
    # save model
    model_orig.save('fashion_mnist_model.h5')
    
    # c)
    # evaluate model on test and augmented sets
    print('Model trained on original training data:')
    evals = model_orig.evaluate(X_test, y_test, verbose=0)
    print('Accuracy Test Set: ' + str(evals[1]))
    evals = model_orig.evaluate(X_test_pret, y_test_pret, verbose=0)
    print('Accuracy Pretend Set: ' + str(evals[1]))
    
    # d)
    model_pret = buildModel()
    model_pret.fit(X_train_pret, y_train_pret, epochs=epochs, verbose=0)
    # evaluate model on augmented set
    print('Model trained on pretend training data:')
    evals = model_pret.evaluate(X_test_pret, y_test_pret, verbose=0)
    print('Accuracy Pretend Set: ' + str(evals[1]))
    
    # e)
    model_aug = buildModel()
    model_aug.fit_generator(data_generator, steps_per_epoch=4000, epochs=1, verbose=0, shuffle=True)
    # evaluate model on test and augmented sets
    print('Model trained with data augmentation:')
    evals = model_aug.evaluate(X_test_pret, y_test_pret, verbose=0)
    print('Accuracy Pretend Set: ' + str(evals[1]))
    
    K.clear_session()