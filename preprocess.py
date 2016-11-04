import smart_open
import re
import os
from time import time

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

    tweet is the cleaned version of input with hashtags removed
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
    def __init__(self, source):
        self.source = source

    def __iter__(self):
        with smart_open.smart_open(self.source) as f:
            for line in f:
                yield splitHashtags(line.decode('utf-8'))


if __name__ == '__main__':
    test_tweets = ['RT @realDonaldTrump: A Clinton economy = more taxes and more spending! #DebateNight https://t.co/oFlaAhrwe5', 'RT @NimbleNavgater: Literally TENS of people showed up to see Hillary and Tim Kaine today in PA! #WeHateHillary #CrookedHillary https://t.c…', 'RT @AP: Nielsen estimates Clinton speech watched by 29.8 million people; 32.2 million watched Trump at RNC. https://t.co/S5CtwXj29A', '#FreeLeonardPelter @BarackObama @POTUS Please do the right thing. Let him spend his last days at home. https://t.co/b4DCFy78mi']

    def print_th(t, h):
        print('tweet: {}\n#hash: {}\n'.format(t, ' '.join(h)))

    sample = os.path.expanduser('~/data/twitter/ece901/sample.csv')

    hash_counts = {i: 0 for i in range(6)}

    print()
    if os.path.exists(sample):
        t0 = time()
        n = 0
        for t, h in TweetIterator(sample):
            if len(h) < 5:
                hash_counts[len(h)] += 1
            else:
                hash_counts[5] += 1
            if n < 10:
                print_th(t, h)
            n += 1
        t = time() - t0
        print('Processing {} tweets took {}s...\nWould take ~{} minutes for 1million tweets...'.format(n, t, 10**6 * t / (60 * n)))
    else:
        for t, h in splitHashtags(test_tweets):
            if len(h) < 5:
                hash_counts[len(h)] += 1
            else:
                hash_counts[5] += 1
            print_th(t, h)

    print('Count by # of hashtags:')
    for h, c in hash_counts.items():
        print('{}: {}'.format(h, c))
    print()
