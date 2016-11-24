#!/usr/bin/python

from smart_open import smart_open
import re
import os
from sklearn.preprocessing import MultiLabelBinarizer
from utils import savePickle
from utils import loadPickle
from utils import loadWord2Vec
import sys
from warnings import warn
import string
import numpy as np
from itertools import cycle


# Try to load the word2vec model and the multilabelbinarizer
w2vfile = './models/w2v'
mlbfile = './models/mlb.pickle'
w2v = False

# Loading pickle files is faster, so check that one first
if os.path.exists(w2vfile + '.pickle'):
    w2v = loadPickle(w2vfile + '.pickle')
elif os.path.exists(w2vfile + '.bin'):
    w2v = loadWord2Vec(w2vfile + '.bin')
else:
    warn("{} not found, will not be able to sub or create word matrices".format(w2vfile))

if w2v:
    word_d = w2v.layer1_size

if os.path.exists(mlbfile):
    mlb = loadPickle(mlbfile)
    valid_hashtags = set(mlb.classes_)
else:
    valid_hashtags = set()
    warn("{} not found, will not be able to encode hashtags as vectors".format(mlbfile))

# Compiling some regular expressions for speed
hashtag_regex = re.compile(r'\A#\w+|\s#\w+')
mention_regex = re.compile(r'\A@\w+|\s@\w+')
email_regex = re.compile(r'\S+@\S+.\S+')
url_regex = re.compile(r'\Ahttp\S+|\shttp\S+')
nonprintable_regex = re.compile(r'[^ -~]+')
retweet_regex = re.compile(r'\Art\s|\srt\s')
noncharacter_regex = re.compile(r'[^\s@0-9a-zA-Z]+')
multspace_regex = re.compile(r'\s+')
char_options = string.ascii_lowercase + string.digits + string.punctuation
char_options_set = set(char_options)


class TweetIterator:
    '''
    This is the main preprocessing generator class
    You can iterate through an instance of this

    Inputs:
        source: Source of text/tweets. Can be a list of strings or a file,
            one tweet per line. (newline characters must be stripped)
        skip_nohashtag: if True, iterator will skip tweets with no hashtags.
            If a MultiLabelBinarizer is loaded, this wil only take into account
            hashtags known by this model.
        yield_list:
            What iterator should yield. Can be any of:
                'hashtags', 'raw_tweet', 'raw_tweet_nohashtags', 'tokenized_tweet', 'clean_tweet', 'word_mat', 'chrd_mat', 'char_mat', 'label'

    Usage:

    To simply print a cleaned version of your tweets:

        tweet_iterator = TweetIterator('tweet_file.txt', False, 'clean_tweet')
        for tweet in ti:
            print(tweet)


    Or to get word-embedding matrix and label for each tweet
    (skipping over ones without hashtags in your MultiLabelBinarizer)

        tweet_iterator = TweetIterator('tweet_file.txt', True, 'word_mat', 'label')
        keras_model.fit_generator(tweet_iterator)

    '''
    def __init__(self, source, skip_nohashtag, *yield_list):
        self.source = source
        self.yield_list = []
        yw_options = {'hashtags', 'raw_tweet', 'raw_tweet_nohashtags', 'tokenized_tweet', 'clean_tweet', 'word_mat', 'char_mat', 'chrd_mat', 'label'}
        for yw in yield_list:
            if yw in yw_options:
                self.yield_list.append(yw)
            else:
                warn("Can't yield {}, will skip when iterating".format(yw))
        if len(self.yield_list) == 0:
            warn("No valid options, this iterator won't yield anything")
        self.skip_nohashtag = skip_nohashtag
        self.iter_ = self.__iter__()

    def yield_(self, text):
        tweet, hashtags = split_hashtags(text)
        if valid_hashtags:
            hashtags = [h for h in hashtags if h in valid_hashtags]
        if len(hashtags) == 0 and self.skip_nohashtag:
            return []
        out = []
        for yw in self.yield_list:
            if yw == 'hashtags':
                out.append(hashtags)
            elif yw == 'raw_tweet':
                out.append(text.strip())
            elif yw == 'raw_tweet_nohashtags':
                out.append(tweet)
            elif yw == 'clean_tweet':
                out.append(clean(tweet))
            elif yw == 'tokenized_tweet':
                out.append(clean(tweet).split())
            elif yw == 'word_mat':
                out.append(text2mat(tweet, mat_type='word'))
            elif yw == 'char_mat':
                out.append(text2mat(tweet, mat_type='char'))
            elif yw == 'chrd_mat':
                out.append(text2mat(tweet, mat_type='chrd'))
            elif yw == 'label':
                out.append(mlb.transform([hashtags]))
        return out

    def __iter__(self):
        if type(self.source) == list:
            for text in self.source:
                yw = self.yield_(text)
                if len(yw) == 0:
                    continue
                elif len(yw) == 1:
                    yw = yw[0]
                yield yw
        else:
            with smart_open(self.source, 'r') as f:
                for text in f:
                    yw = self.yield_(text)
                    if len(yw) == 0:
                        continue
                    elif len(yw) == 1:
                        yw = yw[0]
                    yield yw

    def __next__(self):
        return self.next()

    def next(self):
        return next(self.iter_)


class KerasIterator:
    '''
    The iterator class that iterates through source and feeds features/labels into our keras model.
    Usage:
        model.fit_generator(KerasIterator('source.txt'))

    As the keras model requires, THIS ITERATES FOREVER, so it is not recommended you use this class for other purposes.
    '''
    def __init__(self, source, batch_size=10, char=True, chrd=True, word=True):
        if not (char or chrd or word):
            warn("No matrix type specified, this KerasIterator probably won't work right")
        mat_types = []
        if char:
            mat_types.append('char_mat')
        if chrd:
            mat_types.append('chrd_mat')
        if word:
            mat_types.append('word_mat')
        mat_types.append('label')
        tweet_iterator = TweetIterator(source, True, *mat_types)
        self.char = char
        self.chrd = chrd
        self.word = word
        self.iter = cycle(tweet_iterator)
        self.batch_size = batch_size
        self.iter_ = self.__iter__()

    def __iter__(self):
        output_charX = []
        output_chrdX = []
        output_wordX = []
        output_y = []
        i = 0
        for outs in self.iter:
            argout = 0
            if self.char:
                output_charX.append(outs[argout])
                argout += 1
            if self.chrd:
                output_chrdX.append(outs[argout])
                argout += 1
            if self.word:
                output_wordX.append(outs[argout])
                argout += 1
            output_y.append(outs[argout])
            i += 1
            if i == self.batch_size:
                out = []
                if self.char:
                    out.append(np.stack(output_charX))
                if self.chrd:
                    out.append(np.stack(output_chrdX))
                if self.word:
                    out.append(np.stack(output_wordX))
                yield out, np.vstack(output_y)
                i = 0
                output_charX = []
                output_chrdX = []
                output_wordX = []
                output_y = []

    def __next__(self):
        return self.next()

    def next(self):
        return next(self.iter_)


def text2mat(text, mat_type='char', max_chars=140, max_words=50):
    if mat_type == 'char':
        M = np.zeros((max_chars, len(char_options)))
        for i, c in enumerate(text.lower()):
            if i >= max_chars:
                break
            if c in char_options_set:
                c_pos = char_options.index(c)
                M[i, c_pos] = 1
    elif mat_type == 'word':
        text = clean(text)
        M = np.zeros((max_words, word_d))
        i = 0
        for word in text.split():
            if i >= max_words:
                break
            if word in w2v:
                M[i, :] = w2v[word]
                i += 1
    elif mat_type == 'chrd':
        M = np.zeros((max_words, len(char_options)))
        for i, word in enumerate(text.lower().split()):
            if i >= max_words:
                break
            for c in word:
                if c in char_options_set:
                    c_pos = char_options.index(c)
                    M[i, c_pos] += 1

    return M


def split_hashtags(tweet):
    '''
    Returns tweet, hashtags

    tweet with hashtags removed
    hashtags is a list of #hashtags
    '''
    if type(tweet) == list:
        return [split_hashtags(t) for t in tweet]
    hashtags = hashtag_regex.findall(tweet)
    hashtags = [h.strip().lower() for h in hashtags]
    tweet = hashtag_regex.sub('', tweet)
    return tweet, hashtags


def clean(tweet):
    '''
    Cleans tweet for our model
    '''
    # TODO this screws up urls (I think I fixed that?) and generates hashtags that are a bunch of spaces for some reason
    if type(tweet) == list:
        return [clean(t) for t in tweet]
    tweet = url_regex.sub(' http://url', tweet)
    tweet = noncharacter_regex.sub(' ', tweet)
    tweet = email_regex.sub(' email@address ', tweet)
    tweet = mention_regex.sub(' @user', tweet)
    tweet = noncharacter_regex.sub('', tweet)
    tweet = tweet.lower()
    tweet = retweet_regex.sub(' ', tweet)
    tweet = multspace_regex.sub(' ', tweet)
    tweet = tweet.strip()
    return tweet


def sub(tweet, thresh=.9):
    '''
    Uses word2vec model to produce a similar version of tweet,
    for augmenting our model.

    Still in progress.
    '''
    # TODO cache "most_similar" for speed?
    # Or just break when it's done?
    # Or just pick random words to try to sub?
    words = tweet.split()
    for i in range(len(words)):
        if words[i] in w2v:
            most_sim = w2v.most_similar(words[i])[0]
            if most_sim[1] > thresh:
                words[i] = most_sim[0]
                break

    return ' '.join(words)


def PrepareMLB(source, threshold=50):
    '''
    This function produces the "MultiLabelBinarizer" object and saves it as a
    pickle file in the ./models directory

    The MultiLabelBinarizer is the object that turns a list of hashtags into a
    sparse binary vector, for labels for our model

    This function will filter out any hashtags that appear fewer than `threshold` times
    '''

    model_dir = './models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    output_mlb = os.path.join(model_dir, 'mlb.pickle')

    final_set = set()
    counts = {}

    for hashtags in TweetIterator(source, False, 'hashtags'):
        for h in hashtags:
            if h not in final_set:
                if h not in counts:
                    counts[h] = 1
                else:
                    counts[h] += 1
                    if counts[h] >= threshold:
                        final_set.add(h)
                        del counts[h]

    mlb = MultiLabelBinarizer(sparse_output=False).fit([list(final_set)])
    savePickle(mlb, output_mlb)


def Test(source, skip=False):
    print("\nThis should print the raw text of your tweets:\n")
    for i in TweetIterator(source, skip, 'raw_tweet'):
        print(i)

    print("\nThis should print the clean text of your tweets:\n")
    for i in TweetIterator(source, skip, 'clean_tweet'):
        print(i)

    print("\nThis should print the hashtags of your tweets:\n")
    for i in TweetIterator(source, skip, 'hashtags'):
        print(i)
    print()

    print("\nThis should print the character matrix embedding of your tweets:\n")
    for i in TweetIterator(source, skip, 'char_mat'):
        print(i.shape)
    print()

    print("\nThis should print the label vector of your tweets:\n")
    for i in TweetIterator(source, skip, 'label'):
        print(i.shape)
    print()


if __name__ == '__main__':
    test_tweets = ['RT @realDonaldTrump: A Clinton economy = more taxes and more spending! #DebateNight https://t.co/oFlaAhrwe5', 'RT @NimbleNavgater: Literally TENS of people showed up to see Hillary and Tim Kaine today in PA! #WeHateHillary #CrookedHillary https://t.c…', 'RT @AP: Nielsen estimates Clinton speech watched by 29.8 million people; 32.2 million watched Trump at RNC. https://t.co/S5CtwXj29A', '#FreeLeonardPelter @BarackObama @POTUS Please do the right thing. Let him spend his last days at home. https://t.co/b4DCFy78mi']

    last_arg = sys.argv[-1]
    ext = os.path.splitext(last_arg)[-1]
    valid_exts = ['.txt', '.csv']

    threshold = 10

    if ext in valid_exts:
        sample = sys.argv[-1]
    else:
        sample = './data/sample.csv'

    if not os.path.exists(sample):
        print("\n'{}' doesn't exist, running script on a few test tweets instead.".format(sample))
        sample = test_tweets
        threshold = 0

    if '--threshold' in sys.argv:
        threshold = int(sys.argv[sys.argv.index('--threshold') + 1])
    elif '-t' in sys.argv:
        threshold = int(sys.argv[sys.argv.index('-t') + 1])

    if '--prepare' in sys.argv or '-p' in sys.argv:
        PrepareMLB(sample, threshold=threshold)
    else:
        Test(sample, True)
