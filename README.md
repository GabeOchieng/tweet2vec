# tweet2vec
THE project. 

## Preprocessing
To run preprocessing:
```
python preprocess.py data.txt
```
This will iterate through the lines of `data.txt`, clean the tweets and produce encoded vectors for the hashtags. If run without arguments (i.e. `python preprocess`), will default to looking for a file called `./data/sample.csv` and if not found will jsut run on a few test tweets.
