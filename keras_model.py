from keras.models import Sequential
from keras.layers import Merge
from keras.layers import GRU
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import SGD
from keras.optimizers import RMSprop
# from keras.layers.convolutional import Convolution1D
# from keras.layers.pooling import GlobalAveragePooling1D
from keras.layers.wrappers import Bidirectional
from keras import backend as K
from preprocess import KerasIterator
from preprocess import TweetIterator
from preprocess import text2mat
from utils import loadPickle
import numpy as np
import os
from warnings import warn
from numpy.linalg import norm
from utils import saveTweet2Vec
from utils import loadTweet2Vec
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger


mlb_file = './models/mlb.pickle'
if os.path.exists(mlb_file):
    mlb = loadPickle(mlb_file)
else:
    warn("{} doesn't exist - need this to generate labels for training: run `./preprocess.py --prepare input.txt` first")


class Tweet2Vec:
    def __init__(self, model=None, char=True, chrd=True, word=True):
        '''
        Initialize stuff
        '''
        self.char = char
        self.chrd = chrd
        self.word = word
        charX, chrdX, wordX, y = next(TweetIterator(['this is to figure out input/output dimensions'], False, 'char_mat', 'chrd_mat', 'word_mat', 'label'))
        self.char_dim = charX.shape[1]
        self.chrd_dim = chrdX.shape[1]
        self.word_dim = wordX.shape[1]
        self.output_dim = y.shape[1]
        self.vector_cache_ = {}

        # I think this just affects the feature preprocessing, probably should be num cores - 1
        # ACTUALLY doesn't seem to do what we want to do - seems to make it so we loop over same data and overfit like crazy
        self.num_workers = 1

        if model is None:
            # nothing specified, generate a model
            self.gen_model()
        elif isinstance(model, str):
            # model is a filename
            self.load(model)
        else:
            # model is a keras model
            self.model = model

        # Former is using a merged model, latter if not
        if hasattr(self.model.layers[0], 'layers'):
            self.get_vec_ = K.function([layer.input for layer in self.model.layers[0].layers] + [K.learning_phase()], [self.model.layers[-2].output])
            num_expected = len([layer.input for layer in self.model.layers[0].layers])
        else:
            self.get_vec_ = K.function([self.model.layers[0].input, K.learning_phase()], [self.model.layers[-2].output])
            num_expected = 1

        num_actual = len([i for i in [char, chrd, word] if i])

        if num_expected != num_actual:
            warn("Number of expected inputs to your model ({}) and number of actual inputs ({}) are different. Either you need to change your model or change the Tweet2Vec() arguments".format(num_expected, num_actual))

    def gen_model(self):
        '''
        Build the model
        '''

        # word matrix branch
        word_branch = Sequential()
        word_branch.add(Bidirectional(GRU(self.word_dim * 4, input_dim=self.word_dim, return_sequences=False), input_shape=(None, self.word_dim)))
        # word_branch.add(GlobalAveragePooling1D())

        # chrd matrix branch
        chrd_branch = Sequential()
        chrd_branch.add(Bidirectional(GRU(self.chrd_dim * 4, input_dim=self.chrd_dim, return_sequences=False), input_shape=(None, self.chrd_dim)))

        # merge models (concat outputs)
        self.model = Sequential()
        # The order here determines the order of your inputs. This must correspond to the standard (char, chrd, word) order.
        merged = Merge([chrd_branch, word_branch], mode='concat')
        self.model.add(merged)
        # self.model = word_branch

        # final hidden layer(s)
        self.model.add(Dropout(.25))
        self.model.add(Dense(3000, activation='relu'))
        self.model.add(Dropout(.5))
        self.model.add(Dense(300))

        # output layer
        self.model.add(Dense(self.output_dim, activation='softmax'))

        # loss function/optimizer
        # sgd = SGD(lr=.25, decay=.05)
        rmsprop = RMSprop(lr=.0025, decay=.05)
        self.model.compile(loss='categorical_crossentropy', optimizer=rmsprop)

    def fit(self, source, test=None, batch_size=100, samples=None, num_epochs=1, checkpoint=False):
        '''
        Fit the model using data in source

        The inputs mean what they mean

        This will loop through source forever so it's okay if the numbers are more than your actual data

        For KerasIterator object, specify what matrices it should yield (char, chrd, word),
        these must correspond to what inputs the model expects.
        Note: the inputs will always feed to the model in the (char, chrd, word) order.
        '''

        keras_iterator = KerasIterator(source, batch_size, char=self.char, chrd=self.chrd, word=self.word)
        if test is None:
            test_iterator = None
            test_length = None
            # If no test, need to monitor train loss not val_loss
            checker = ModelCheckpoint('./models/latest_model.keras', monitor='loss', verbose=1, save_best_only=True)
        else:
            test_iterator = KerasIterator(test, batch_size, char=self.char, chrd=self.chrd, word=self.word)
            # test_length = len(test_iterator.tweet_iterator)
            # TODO debugging
            test_length = 1000
            checker = ModelCheckpoint('./models/latest_model.keras', verbose=1, save_best_only=True)
        if checkpoint:
            callbacks = [checker]
        else:
            callbacks = []

        logger = CSVLogger('./models/epoch_history.csv')
        callbacks.append(logger)

        # If not specified, train on ALL data in source
        if samples is None:
            samples = len(keras_iterator.tweet_iterator)
        self.fit_data = self.model.fit_generator(keras_iterator, samples, num_epochs, validation_data=test_iterator, nb_val_samples=test_length, verbose=1, nb_worker=self.num_workers, pickle_safe=False, callbacks=callbacks)

    def plot(self, filename='./models/training_loss.png'):
        plt.figure()
        plt.plot(self.fit_data.history['loss'], lw=3, label='train', color='r')
        if 'val_loss' in self.fit_data.history:
            plt.plot(self.fit_data.history['val_loss'], lw=3, label='test', color='b')
        plt.title('Model Loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend()
        plt.savefig(filename)

    def evaluate(self, source, batch_size=100):
        '''
        Prints the loss on tweets in source
        '''
        keras_iterator = KerasIterator(source, batch_size, char=self.char, chrd=self.chrd, word=self.word)
        num_samples = len(keras_iterator.tweet_iterator)
        loss = self.model.evaluate_generator(keras_iterator, num_samples, nb_worker=self.num_workers, pickle_safe=False)
        print("\nLoss on the {} samples in {} is: {}\n".format(num_samples, source, loss))

    def predict_hashtags(self, source, num_to_validate=None, num_best=1, batch_size=500):
        '''
        Prints the `num_best` top predicted hashtags for `num_to_validate` lines in source
        '''

        raw = TweetIterator(source, True, 'raw_tweet')

        # If not specified, run on ALL tweets in source
        if num_to_validate is None:
            num_to_validate = len(raw)

        x = self.model.predict_generator(KerasIterator(source, batch_size, char=self.char, chrd=self.chrd, word=self.word), num_to_validate, nb_worker=self.num_workers, pickle_safe=False)

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

        Also works on lists of tweets
        (in fact, this is the recommended way if you are getting lots of vectors
        because it seems to be just as slow getting one vector as it is many)

        caches vectors, so if you ask for them again it will happen in O(1) time
        '''
        if type(tweet) == str:
            tweet = [tweet]

        not_cached = [t for t in tweet if t not in self.vector_cache_]

        # TODO this should happen in batches to avoid memory error
        if not_cached:
            mats_in = []
            if self.char:
                charX = []
                for t in not_cached:
                    charX.append(text2mat(t, 'char'))
                mats_in.append(np.stack(charX))
            if self.chrd:
                chrdX = []
                for t in not_cached:
                    chrdX.append(text2mat(t, 'chrd'))
                mats_in.append(np.stack(chrdX))
            if self.word:
                wordX = []
                for t in not_cached:
                    wordX.append(text2mat(t, 'word'))
                mats_in.append(np.stack(wordX))
            not_cached_vectors = self.get_vec_(mats_in + [0])[0]
            for t, v in zip(not_cached, not_cached_vectors):
                norm_v = norm(v)
                if norm_v == 0:
                    norm_v = 1
                self.vector_cache_[t] = v / norm(v)

        return np.array([self.vector_cache_[t] for t in tweet])

    def most_similar(self, tweet, source, batch_size=500):
        '''
        Iterates through `source` and finds the line with the highest cosine
        similarity to `tweet`
        '''

        best_d = -1
        best_t = ''
        target_v = self[tweet]

        batch = []
        i = 0
        for t in TweetIterator(source, False, 'raw_tweet'):
            batch.append(t)
            i += 1
            if i == batch_size:
                dists = np.dot(self[batch], target_v.T)
                best_i = np.argmax(dists)
                d = dists[best_i]
                t = batch[best_i]
                if d > best_d:
                    best_t = t
                    best_d = d
                i = 0
                batch = []
        if batch:
            dists = np.dot(self[batch], target_v.T)
            best_i = np.argmax(dists)
            d = dists[best_i]
            t = batch[best_i]
            if d > best_d:
                best_t = t
                best_d = d

        return best_t, best_d

    def most_similar_test(self, source1, source2, num_test=10):
        '''
        Another "sanity check"

        Picks a random tweet in source1 and finds the closest tweet in source2 to it
        Does so `num_test` times

        Ideally there is no overlap between the two sources
        '''

        ti = TweetIterator(source1, False, 'raw_tweet')
        for _ in range(num_test):
            t1 = ti.get_random()
            t2, d = self.most_similar(t1, source2)
            print("\nOriginal tweet: {}\nClosest tweet: {}\nDistance: {}\n".format(t1, t2, d))

    def save(self, filename):
        saveTweet2Vec(self.model, filename)

    def load(self, filename):
        self.model = loadTweet2Vec(filename)


if __name__ == '__main__':

    tweet2vec = Tweet2Vec(char=False, chrd=True, word=True)

    train = './data/train.csv'
    test = './data/test.csv'

    train = './data/all_shuffled_train.csv'
    test = './data/all_shuffled_test.csv'

    # samples=None (the default) will train on all input data
    # 6331717 samples in train set
    tweet2vec.fit(train, test=test, samples=10**6, num_epochs=1000, checkpoint=True)
    # tweet2vec.evaluate(test)
    # tweet2vec.most_similar_test(train, test)
    # tweet2vec.plot()
