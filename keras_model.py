from keras.models import Sequential
from keras.layers import GRU
from keras.layers import Dense
from preprocess import KerasIterator
from preprocess import TweetIterator
from utils import loadPickle
import numpy as np
import os
from warnings import warn

mlb_file = './models/mlb.pickle'
if os.path.exists(mlb_file):
    mlb = loadPickle(mlb_file)
else:
    warn("{} doesn't exist - need this to generate labels for training: run `./preprocess.py --prepare input.txt` first")


class Tweet2Vec:
    def __init__(self, input_type='char_mat'):
        self.input_type = input_type
        X, y = next(TweetIterator(['this is to figure out input/output dimensions'], False, self.input_type, 'label'))
        self.input_dim = X.shape[1]
        self.output_dim = y.shape[1]

        self.gen_model()

    def gen_model(self):

        self.model = Sequential()

        # Input layer: # of GRU units
        self.model.add(GRU(100, input_dim=self.input_dim))

        # Hidden layers - go nuts
        self.model.add(Dense(80, activation='relu'))
        self.model.add(Dense(40, activation='relu'))
        self.model.add(Dense(20, activation='relu'))
        self.model.add(Dense(20, activation='relu'))
        self.model.add(Dense(20, activation='relu'))
        self.model.add(Dense(20, activation='relu'))

        # Output layer - maybe change activation?
        self.model.add(Dense(self.output_dim, activation='softmax'))

        # Adjust loss function/optimizer here
        self.model.compile(loss='categorical_crossentropy', optimizer='adam')

    def fit(self, source, batch_size=10, samplers_per_epoch=1000, num_epochs=10):

        keras_iterator = KerasIterator(source, batch_size, self.input_type)
        self.model.fit_generator(keras_iterator, samplers_per_epoch, num_epochs, verbose=1)

    def validate(self, source, num_to_validate=10):

        x = self.model.predict_generator(KerasIterator(source, 1, 'char_mat'), num_to_validate)
        raw = TweetIterator(source, True, 'raw_tweet')

        for i, r in zip(x, raw):
            label = np.zeros((1, i.shape[0]))
            label[0, i.argmax()] = 1
            predicted_hashtag = mlb.inverse_transform(label)[0][0]
            print("\nTweet: {}\nPredicted hashtag: {}\n".format(r, predicted_hashtag))


if __name__ == '__main__':
    tweet2vec = Tweet2Vec()

    train = './data/train.csv'
    test = './data/test.csv'

    tweet2vec.fit(train)
    tweet2vec.validate(test)
