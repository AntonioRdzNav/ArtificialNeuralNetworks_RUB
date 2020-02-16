from keras.callbacks import Callback
import sys
import random
import numpy as np

class PrintCallback(Callback) :
    
    def __init__(self, text, chars, int2char, char2int, model) :
        Callback.__init__(self)
        self.text     = text
        self.chars    = chars
        self.int2char = int2char
        self.char2int = char2int
        self.model    = model
        
    def sample(self, preds, temp=1.0):

        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temp
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probs = np.random.multinomial(1, preds, 1)
        return np.argmax(probs)
    
    def on_epoch_end(self,epoch,logs=None):
        len_seq = 50
        print()
        print('-- Sample text after epoch:', epoch)
    
        start_index = random.randint(0, len(self.text) - len_seq - 1)
        diversity = 0.5

        generated = ''
        sentence = self.text[start_index: start_index + len_seq]
        generated += sentence
        print('Seed : ' + sentence)
        sys.stdout.write(generated)

        for i in range(500):
            x_pred = np.zeros((1, len_seq, len(self.chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, self.char2int[char]] = 1.

            preds = self.model.predict(x_pred, verbose=0)[0]
            next_index = self.sample(preds, diversity)
            next_char = self.int2char[next_index]

            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()
    
    
