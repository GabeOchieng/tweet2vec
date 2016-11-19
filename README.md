# tweet2vec
THE project. 

## Preprocessing
To run preprocessing:
```
python preprocess.py data.txt
```

This will run a test which will iterate a few times through the lines of `data.txt`, and print the raw text, clean text, and hashtags.

Running without arguments (i.e. just `python preprocess.py`) will default to looking for the file `./data/sample.csv`

If the file is not found, it will default to just running on 4 sample tweets provided in the code.

To prepare the data for the model:
```
python preprocess.py --prepare data.txt
```
This will generate the file in `./models/mlb.pickle`, which is a `MultiLabelBinarizer` object. This is the object that turns a list of hashtags into an encoded vector (one-hot for each hashtag), for our model.

By default it will filter out hashtags that don't appear more than 10 times. To change this number to say, 100:
```
python preprocess.py --prepare --threshold 100 data.txt
```

Note: you will want to remove the `./models/mlb.pickle` file if you are generating a new one.
If a `mlb.pickle` exists, the script will load it and then filter out any hashtags that aren't already in the model.


## TweetIterator
The main object in prepocessing.py is the `TweetIterator` object. It allows you to iterator through lines in a text file and yield any number of things useful for our model. This allows you to process and generate features for input files in a memory efficient way (i.e. you never have to load all of your data into memory).

To iterate through the raw text:
```
tweet_iterator = TweetIterator('source.txt', False, 'raw_text')
for t in tweet_iterator:
    print(t)
```

To iterate through the hashtags:
```
tweet_iterator = TweetIterator('source.txt', False, 'hashtags')
for t in tweet_iterator:
    print(t)
```

If you have an `MultilabelBinarizer` object prepared, iterate through the label vectors:
```
tweet_iterator = TweetIterator('source.txt', False, 'labels')
for t in tweet_iterator:
    print(t)
```

Iterate through character matrices:
```
tweet_iterator = TweetIterator('source.txt', False, 'char_mat')
for t in tweet_iterator:
    print(t)
```

If you have a word2vec model saved as `./models/w2v.pickle`, you can iterate through word2vec matrices:
```
tweet_iterator = TweetIterator('source.txt', False, 'word_mat')
for t in tweet_iterator:
    print(t)
```

You can combine any number of these:
```
tweet_iterator = TweetIterator('source.txt', False, 'raw_text', 'clean_test', 'hashtags', 'char_mat', 'word_mat')
for rt, ct, h, cm, wm in tweet_iterator:
    print(rt, ct, h, cm, wm)
```

etc...
