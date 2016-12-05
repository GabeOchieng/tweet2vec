from keras_model import Tweet2Vec
from time import time
import pandas as pd
from utils import savePickle
from utils import loadPickle
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.cluster import spectral_clustering
from sklearn.cluster import k_means

model_file = './models/161205_sgd_lab/latest_model.keras'
source_file = './data/trump_sample.csv'

def get_vecs():

    t0 = time()
    tweet2vec = Tweet2Vec(model_file, char=False, chrd=True, word=True)
    print("Loading model took {}s".format(time() - t0))

    source = pd.read_csv(source_file, header=None, sep=chr(1))
    text = source[0]

    t0 = time()
    M = tweet2vec[text]
    print(M)
    print(M.shape)
    print("Grabbing {} vectors took {}s".format(len(text), time() - t0))

    savePickle(M, './models/trump_sample_vectors.pickle')

def get_affinity():
    t0 = time()
    A = rbf_kernel(loadPickle('./models/trump_sample_vectors.pickle'))
    savePickle(A, './models/trump_sample_affinity.pickle')
    print(A.shape)
    print("Spectral clustering took {}s".format(time() - t0))


def spectral_cluster():
    t0 = time()
    S = spectral_clustering(loadPickle('./models/trump_sample_affinity.pickle'), n_clusters=100)
    savePickle(S, './models/trump_sample_spectral.pickle')
    print(S)
    print("Spectral clustering took {}s".format(time() - t0))

def kmeans():
    t0 = time()
    K = k_means(loadPickle('./models/trump_sample_vectors.pickle'), n_clusters=100, n_jobs=-1)
    savePickle(K, './models/trump_sample_kmeans.pickle')
    print(K)
    print("K-means took {}s".format(time() - t0))

if __name__ == '__main__':
    get_vecs()
    kmeans()
