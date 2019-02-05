import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from joblib import dump, load
import preprocessor


class Filter:
    def __init__(self):
        pass


    def message_process(self,message):
        remove_punc = [char for char in message if char not in string.punctuation]
        no_punc = ''.join(remove_punc)
        # tokens = [token for token in tokens if token.lower() not in stopwords.words("english")]

        return ''.join(no_punc)


if __name__=='__main__':

    sentence='it is working?,,:'
    filter=Filter()
    x=filter.message_process(sentence)
    print(x)