from utils import loadPickle
from smart_open import smart_open
from preprocess import TweetHashtagIterator

w2vfile = './models/w2v_1day.pickle'
w2v = loadPickle(w2vfile)


class TweetSubIterator(TweetHashtagIterator):
    def __init__(self, source):
        TweetHashtagIterator.__init__(self, source, 'tweet', True)

    def __iter__(self):
        pass


def sub(tweet, thresh=.9):
    # TODO cache "most_similar" for speed?
    words = tweet.split()
    most_sims = []
    for word in words:
        if word in w2v:
            most_sim = w2v.most_similar(word)[0]
            if most_sim[1] > thresh:
                most_sims.append(most_sim[0])
            else:
                most_sims.append(word)
        else:
            most_sims.append(word)

    return ' '.join(most_sims)


def main():

    test_tweets = ['@user a clinton economy  more taxes and more spending httpurl', 
                   '@user literally tens of people showed up to see hillary and tim kaine today in pa httpurl',
                   '@user nielsen estimates clinton speech watched by 298 million people 322 million watched trump at rnc httpurl']

    for t in test_tweets:
        print("Original: {}".format(t))
        for thresh in [.9, .85, .8, .75, .7]:
            print("Sub: {}".format(sub(t, thresh)))


if __name__ == '__main__':
    main()
