import pickle


def savePickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def loadPickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
