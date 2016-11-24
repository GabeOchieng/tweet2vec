from keras.models import Sequential
from keras.layers import Merge
from keras.layers import GRU
from keras.layers import Dense
from keras.layers.wrappers import Bidirectional
from keras import backend
from preprocess import KerasIterator
from preprocess import TweetIterator
from preprocess import text2mat
from utils import loadPickle
import numpy as np
import os
from warnings import warn
from scipy.spatial.distance import cosine as cosine_distance
from time import time


mlb_file = './models/mlb.pickle'
if os.path.exists(mlb_file):
    mlb = loadPickle(mlb_file)
else:
    warn("{} doesn't exist - need this to generate labels for training: run `./preprocess.py --prepare input.txt` first")


# TODO need saving/loading models. There's already a function in utils to do this but haven't looked into the details
# TODO need a consistent way to determine what are the inputs/what KerasIterator should iterate over... specify when generating model?  Creating Tweet2Vec object? Fitting?


class Tweet2Vec:
    def __init__(self, model=None):
        '''
        Initialize stuff
        '''
        charX, chrdX, wordX, y = next(TweetIterator(['this is to figure out input/output dimensions'], False, 'char_mat', 'chrd_mat', 'word_mat', 'label'))
        self.char_dim = charX.shape[1]
        self.chrd_dim = chrdX.shape[1]
        self.word_dim = wordX.shape[1]
        self.output_dim = y.shape[1]

        if model is None:
            self.gen_model()
        else:
            self.model = model

        self.get_vec_ = backend.function([layer.input for layer in self.model.layers[0].layers], [self.model.layers[-2].output])

    def gen_model(self):
        '''
        Build the model
        '''

        # chrd matrix branch
        chrd_branch = Sequential()
        chrd_branch.add(Bidirectional(GRU(20, input_dim=self.chrd_dim, return_sequences=False), input_shape=(None, self.chrd_dim)))
        chrd_branch.add(Dense(20, activation='relu'))

        # word matrix branch
        word_branch = Sequential()
        word_branch.add(Bidirectional(GRU(20, input_dim=self.word_dim, return_sequences=False), input_shape=(None, self.word_dim)))
        word_branch.add(Dense(20, activation='relu'))

        # merge models (concat outputs)
        self.model = Sequential()

        # The order here determines the order of your inputs. This must correspond to the standard (char, chrd, word) order.
        merged = Merge([chrd_branch, word_branch], mode='concat')
        self.model.add(merged)

        # final hidden layer
        self.model.add(Dense(50, activation='relu'))

        # output layer
        self.model.add(Dense(self.output_dim, activation='softmax'))

        # loss function/optimizer
        self.model.compile(loss='categorical_crossentropy', optimizer='adam')

    def fit(self, source, batch_size=10, samples_per_epoch=1000, num_epochs=100):
        '''
        Fit the model using data in source

        The inputs mean what they mean

        This will loop through source forever so it's okay if the numbers are more than your actual data

        For KerasIterator object, specify what matrices it should yield (char, chrd, word),
        these must correspond to what inputs the model expects.
        Note: the inputs will always feed to the model in the (char, chrd, word) order.
        '''
        keras_iterator = KerasIterator(source, batch_size, char=False, chrd=True, word=True)
        self.model.fit_generator(keras_iterator, samples_per_epoch, num_epochs, verbose=1)

    def validate(self, source, num_to_validate=10, num_best=1):
        '''
        More of a "sanity check"

        Prints top predicted hashtag for `num_to_validate` lines in source
        '''

        x = self.model.predict_generator(KerasIterator(source, 1), num_to_validate)
        raw = TweetIterator(source, True, 'raw_tweet')

        for i, r in zip(x, raw):
            # goes through the highest prediction values and outputs
            if num_best > 1:
                best = i.argsort()[-num_best:][::-1]
            else:
                best = [i.argmax()]

            print("\nTweet: {}".format(r))
            best_hashtags = []
            for b in best:
                label = np.zeros((1, i.shape[0]))
                label[0, b] = 1
                predicted_hashtag = mlb.inverse_transform(label)[0][0]
                best_hashtags.append(predicted_hashtag)
            print("Predicted hashtags: {}\n".format(', '.join(best_hashtags)))

    def __getitem__(self, tweet):
        '''
        Gets the vector for tweet like the word2vec api
        e.g.

            tweet2vec['Raw text of the tweet']

        will return the vector.

        This is slow
        '''
        charX = text2mat(tweet, 'char')
        chrdX = text2mat(tweet, 'chrd')
        wordX = text2mat(tweet, 'word')
        charX = np.expand_dims(charX, 0)
        chrdX = np.expand_dims(chrdX, 0)
        wordX = np.expand_dims(wordX, 0)
        return self.get_vec_([charX, chrdX, wordX])[0][0]

    def most_similar(self, tweet, source):
        '''
        Iterates through source and finds the line with the highest cosine
        similarity to input

        This is very slow
        '''
        # TODO look how gensim's word2vec implements this - it's very fast

        best_d = -1
        best_t = ''
        target_v = self[tweet]

        t0 = time()

        for i, t in enumerate(TweetIterator(source, False, 'raw_tweet')):
            d = cosine_distance(target_v, self[t])
            if d > best_d:
                best_d = d
                best_t = t
            if i % 100 == 0:
                print("At {} with {}s passed".format(i, time() - t0))

        return best_t, best_d


if __name__ == '__main__':
    tweet2vec = Tweet2Vec()

    train = './data/train.csv'
    test = './data/test.csv'

    tweet2vec.fit(train, samples_per_epoch=100, num_epochs=1)
