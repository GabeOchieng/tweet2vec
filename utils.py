import pickle
from gensim.models import word2vec
from keras.models import load_model


def savePickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def loadPickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def saveWord2Vec(model, filename):
    model.init_sims(replace=True)
    model.save_word2vec_format(filename, binary=True)


def loadWord2Vec(filename):
    model = word2vec.Word2Vec()
    return model.load_word2vec_format(filename, binary=True)


def saveTweet2Vec(model, filename):
    model.save(filename)


def loadTweet2Vec(filename):
    return load_model(filename)
