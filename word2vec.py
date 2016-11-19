import gensim.models.word2vec as word2vec
from utils import savePickle
from time import time
from preprocess import TweetHashtagIterator
import multiprocessing


def train():

    data_file = '~/data/twitter/ece901/161112politics0.csv'
    model_file = './models/w2v.pickle'

    data = TweetHashtagIterator(data_file, 'tweet', tokenize=True)

    t0 = time()
    model = word2vec.Word2Vec(data, workers=multiprocessing.cpu_count(), sg=1)
    print("Training word2vec model took {}s".format(time() - t0))

    savePickle(model, model_file)

if __name__ == '__main__':
    train()
