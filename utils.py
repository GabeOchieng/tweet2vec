import pickle
from gensim.models import word2vec
from keras.models import load_model
from itertools import takewhile
from itertools import repeat


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


def saveList(l, filename):
    with open(filename, 'w') as f:
        f.write('\n'.join(l))


def loadList(filename):
    with open(filename) as f:
        l = f.readlines()
    l = [s.strip() for s in l]
    return l


def countLines(filename):
    with open(filename, 'rb') as f:
        bufgen = takewhile(lambda x: x, (f.raw.read(1024 * 1024) for _ in repeat(None)))
        num_lines = sum(buf.count(b'\n') for buf in bufgen)
    return num_lines
