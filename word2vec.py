import gensim.models.word2vec as word2vec
from utils import savePickle
from utils import saveWord2Vec
from time import time
from preprocess import TweetIterator
import multiprocessing


def train():

    data_file = '~/data/twitter/ece901/161112politics0.csv'
    model_base = './models/w2v'
    model_pickle = model_base + '.pickle'
    model_bin = model_base + '.bin'

    data = TweetIterator(data_file, False, 'tokenized_tweet')

    t0 = time()
    model = word2vec.Word2Vec(data, workers=multiprocessing.cpu_count(), sg=1)
    print("Training word2vec model took {}s".format(time() - t0))

    savePickle(model, model_pickle)
    saveWord2Vec(model, model_bin)


if __name__ == '__main__':
    train()
