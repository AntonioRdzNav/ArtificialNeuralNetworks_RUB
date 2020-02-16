from print_callback import PrintCallback
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN, LSTM
from keras.optimizers import Adam
import numpy as np

import io

# data: Raw data.
# input steps : The number of steps to be used as input .
# output : The number of steps to be used as label .  
def prepareData(data, input_steps=1, output=1):
# prepare inputs    
    X = []  
    for example in range(data.shape[0] - input_steps):
        X += [np.reshape ( data [example:example + input_steps ], ( input_steps , 1))]
# prepare labels
    y = []
    for example in range( input_steps , data . shape[0] - output + 1):
        y += [ data [example:example + output ]]
    return np.array(X), np.array(y)    

def buildModel(input_steps, output):
    model = Sequential()
    model.add(SimpleRNN(units=32, input_shape=(input_steps, 1), activation="relu"))
    model.add(Dense(output))
    model.compile(loss="mean_squared_error", optimizer="adam")
    return model
###################################################################################
################################# SOLUTION 1 ######################################
###################################################################################
def predictingTimeSeriesData():
    N = 1000
    time_steps = 1
    output = 1
    N_split = int(N*0.7)
    # generate raw data
    X = np.linspace(0,100, num=N)
    sine = np. sin (X) + X*0.1
    # generate model
    model = buildModel(time_steps, output)
    # prepare data
    X, Y = prepareData(sine, input_steps=time_steps, output=output)
    # split data
    X_train , X_test , Y_train , Y_test = X[: N_split ], X[N_split :], Y[: N_split ], Y[ N_split :]    
    # fit model
    model.fit(X_train, Y_train, batch_size=16, epochs=20, verbose=2)
# predictingTimeSeriesData()






def readText():
#open the file
    path = 'faust.txt'
    with io.open(path, encoding='utf-8') as f:
        text = f.read().lower()
#Look at the text characteristics to get an idea of the text
    print('-----------Text characteristics--------')
    print('Text length:', len(text))
    chars = sorted(list(set(text)))
    print('Number of unique characters in text:', len(chars))    
    return text, chars

def prepareTrainingData(text, len_seq, step):
    sequences = []
    targets = []     # labels are the next character after each sequence
    # (len_seq) is subtracted because last (len_seq) caracters are inside last sequence 
    for i in range(0, len(text) - len_seq, step):
        sequences.append(text[i: i + len_seq])  # grab whole 50 characters
        targets.append(text[i + len_seq])       # get next character
    print('Number of training sequences:', len(sequences))    
    return sequences, targets

def generateOneHotEncoding(sequences, targets, chars, len_seq, char2int):
    x = np.zeros((len(sequences), len_seq, len(chars)))
    y = np.zeros((len(sequences), len(chars)))

    #compute one-hot encoding
    for i, sentence in enumerate(sequences):
        for t, char in enumerate(sentence):
            x[i, t, char2int[char]] = 1
        y[i, char2int[targets[i]]] = 1    
    return x, y

def buildModel2(len_seq, chars):
# cross entropy and softmax are used because this is a classification problem
# the output layer (Dense) should have as much as different characters there are
    model = Sequential()
    model.add(LSTM(256, input_shape=(len_seq, len(chars))))
    model.add(Dense(len(chars), activation='softmax'))
    optimizer = Adam(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    return model
###################################################################################
################################# SOLUTION 2 ######################################
###################################################################################
def learningLongTermDependencies():  
# a)
    text, chars = readText()

# map characters to indices and vice-versa
    char2int = dict((c, i) for i, c in enumerate(chars))
    int2char = dict((i, c) for i, c in enumerate(chars))
# prepare training data
    len_seq = 50
    step = 3
    sequences, targets = prepareTrainingData(text, len_seq, step)

# c) generate one-hot encoding
    x, y = generateOneHotEncoding(sequences, targets, chars, len_seq, char2int)

# d) build model
    model = buildModel2(len_seq, chars)

# e) Train the model
    print_callback = PrintCallback(text, chars, int2char, char2int, model)
    model.fit(x, y,
            batch_size=128,
            epochs=20,
            callbacks=[print_callback])
    model.save_weights('weights_lstm.h5f')

learningLongTermDependencies()    