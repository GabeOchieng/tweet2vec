import smart_open
import re
import os
from sklearn.preprocessing import MultiLabelBinarizer
from utils import savePickle
import numpy as np
import sys


# Compiling some regular expressions for speed
hashtag_regex = re.compile(r'\A#\w+| #\w+')
mention_regex = re.compile(r'\A@\w+| @\w+')
url_regex = re.compile(r'\Ahttp\S+| http\S+')
nonprintable_regex = re.compile(r'[^ -~]+')
retweet_regex = re.compile(r'\Art | rt ')
noncharacter_regex = re.compile(r'[^ @0-9a-zA-Z]+')


def splitHashtags(tweet):
    '''
    Returns tweet, hashtags

    tweet is the cleaned version of input with hashtags removed, etc.
    hashtags is a list of #hashtags
    '''
    if type(tweet) == list:
        return [splitHashtags(t) for t in tweet]
    hashtags = hashtag_regex.findall(tweet)
    hashtags = [h.strip().lower() for h in hashtags]
    tweet = hashtag_regex.sub('', tweet)
    tweet = clean(tweet)
    return tweet, hashtags


def clean(tweet):
    '''
    Cleans tweet for our model
    '''
    if type(tweet) == list:
        return [clean(t) for t in tweet]
    tweet = url_regex.sub(' http://url', tweet)
    tweet = mention_regex.sub(' @user', tweet)
    tweet = noncharacter_regex.sub('', tweet)
    tweet = tweet.lower()
    tweet = retweet_regex.sub(' ', tweet)
    tweet = tweet.strip()
    return tweet


class TweetIterator:
    '''
    An iterator that iterates through source (can be filename or list)
    Yields:
        tweet, hashtag
    where tweet is cleaned tweet, and hashtag is list of hashtags.
    '''
    def __init__(self, source):
        self.source = source

    def __iter__(self):
        if type(self.source) == list:
            for t in self.source:
                yield splitHashtags(t)
        else:
            with smart_open.smart_open(self.source) as f:
                for line in f:
                    yield splitHashtags(line.decode('utf-8'))


def Prepare(source):
    '''
    This is the main preprocessing function.
    Iterates through source, cleans tweets/strips hashtags.
    Saves clean data (plain text) and labels (a sparse CSR matrix) in ./data directory
    Saves the "MultiLabelBinarizer" in ./models directory
    (this is the object that turns a list of hashtags into a binary vector and back)
    '''
    data_dir = './data'
    model_dir = './models'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    output_text = os.path.join(data_dir, 'cleandata.txt')
    output_labels = os.path.join(data_dir, 'labels.pickle')
    output_mlb = os.path.join(model_dir, 'mlb.pickle')
    mlb = MultiLabelBinarizer(sparse_output=True)
    labels = []
    num_tweets = 0
    with smart_open.smart_open(output_text, 'w') as f:
        for tweet, hashtags in TweetIterator(source):
            if num_tweets > 0:
                f.write('\n')
            labels.append(hashtags)
            f.write(tweet)
            num_tweets += 1

    labels = mlb.fit_transform(labels)
    savePickle(labels, output_labels)
    savePickle(mlb, output_mlb)

    print("Processed {} tweets".format(num_tweets))
    i = np.random.choice(labels.shape[0])
    j = 0
    while np.sum(labels[i]) == 0:
        # Find a nontrivial to print, give up after 100 tries...
        i = np.random.choice(labels.shape[0])
        j += 1
        if j > 100:
            break
    print("Label {} looks like:\n{}".format(mlb.inverse_transform(labels[i]), labels[i].todense()))
    print()


def Print(source):
    '''
    Iterates through tweets in source and prints cleaned tweet and extracted hashtags
    Mostly used for testing and debugging the "clean" and "split" steps.
    '''
    def print_th(t, h):
        print('tweet: {}\n#hash: {}\n'.format(t, ' '.join(h)))

    if type(source) == list:
        for t, h in splitHashtags(source):
            print_th(t, h)
    else:
        for t, h in TweetIterator(source):
            print_th(t, h)


if __name__ == '__main__':
    test_tweets = ['RT @realDonaldTrump: A Clinton economy = more taxes and more spending! #DebateNight https://t.co/oFlaAhrwe5', 'RT @NimbleNavgater: Literally TENS of people showed up to see Hillary and Tim Kaine today in PA! #WeHateHillary #CrookedHillary https://t.c…', 'RT @AP: Nielsen estimates Clinton speech watched by 29.8 million people; 32.2 million watched Trump at RNC. https://t.co/S5CtwXj29A', '#FreeLeonardPelter @BarackObama @POTUS Please do the right thing. Let him spend his last days at home. https://t.co/b4DCFy78mi']

    if len(sys.argv) > 1:
        sample = sys.argv[1]
    else:
        sample = './data/sample.csv'

    if os.path.exists(sample):
        Print(sample)
        Prepare(sample)
    else:
        print()
        print("'{}' doesn't exist, running script on a few test tweets instead.".format(sample))
        print()
        Print(test_tweets)
        Prepare(test_tweets)
