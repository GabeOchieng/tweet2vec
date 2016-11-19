# tweet2vec
THE project. 

## Preprocessing
To run preprocessing:
```
python preprocess.py data.txt
```

This will run a test which will iterate throught he lines of `data.txt`, print the raw text, clean text and hashtags.

Running without arguments will look for a file `./data/sample.csv` and if not found will just run on 4 sample tweets.

```
python preprocess.py --prepare data.txt
```

Will the file in `./models/mlb.pickle`, which is a `MultiLabelBinarizer` object which is the object that turns a list of hashtags into an encoded vector (one-hot for each hashtag), for our model. By default it will filter out hashtags that don't appear more than 10 times. To change this number to say, 100:

```
python preprocess.py --prepare --threshold 100 data.txt
```

Note: you will want to remove the `./models/mlb.pickle` file if you are generating another one, because by default the `TweetIterator` object filters out hashtags that don't appear in the `MultiLabelBinarizer` object, if it exists.
