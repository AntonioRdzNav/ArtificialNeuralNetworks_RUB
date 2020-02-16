# basic imports
import pickle
# keras imports
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
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
    # number of epochs to be trained
    epochs = 1
    
    # a)
    # load augmented data set
    data_augmented = pickle.load(open('fashion_mnist_augmented.pkl','rb'))
    X_augmented, y_augmented = data_augmented['X'], data_augmented['y']
    # split augmented data set
    X_train_pret, X_test_pret, y_train_pret, y_test_pret = X_augmented[:10000], X_augmented[10000:], y_augmented[:10000], y_augmented[10000:]
    
    # b)
    # test for first 500 samples
    # load model
    model_orig = load_model('fashion_mnist_model.h5')
    model_orig.fit(X_train_pret[:500], y_train_pret[:500], epochs=epochs, verbose=1)
    eval_orig = model_orig.evaluate(X_test_pret, y_test_pret, verbose=0)
    # train denovo model
    model_novo = buildModel()
    model_novo.fit(X_train_pret[:500], y_train_pret[:500], epochs=epochs, verbose=1)
    eval_novo = model_novo.evaluate(X_test_pret, y_test_pret, verbose=0)
    # evaluate model on test and augmented sets
    print('N=500')
    print('Accuracy Original: ' + str(eval_orig[1]))
    print('Accuracy Novo: ' + str(eval_novo[1]))
    # test for full training set
    # load model
    model_orig = load_model('fashion_mnist_model.h5')
    model_orig.fit(X_train_pret, y_train_pret, epochs=epochs, verbose=0)
    eval_orig = model_orig.evaluate(X_test_pret, y_test_pret, verbose=0)
    # train denovo model
    model_novo = buildModel()
    model_novo.fit(X_train_pret, y_train_pret, epochs=epochs, verbose=0)
    eval_novo = model_novo.evaluate(X_test_pret, y_test_pret, verbose=0)
    # evaluate model on test and augmented sets
    print('N=30000')
    print('Accuracy Original: ' + str(eval_orig[1]))
    print('Accuracy Novo: ' + str(eval_novo[1]))
    
    K.clear_session()