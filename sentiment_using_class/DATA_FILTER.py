from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from joblib import dump, load
import preprocessor


class Filter:
    def __init__(self):
        print('============the dataset is here=========')

    def message_process(self,message):
        remove_punc = [char for char in message if char not in string.punctuation]
        no_punc = ''.join(remove_punc)

        tokens = no_punc.split()
        #tokens = [token for token in tokens if token.lower() not in stopwords.words("english")]

        return ' '.join(tokens)

    def featuring(self,df):
    # features with df['review']
        feature = self.message_process(df['review'])
        #print('the features are=e=========',feature)
        processed_data =df['review'].apply(self.message_process) ###removes all the unwanted words from sentences

        #print('============the training is done here=========')
        return processed_data
    # the data suitable for tf trainer is processed_data (up)
    # print('the vocabulary  are========',processed_data)

    def tokenizing(self,processed_data):
    # Training tf Vectorizer
        tf = TfidfVectorizer(ngram_range=(1, 2))
        #print('the tfs are=e=========',tf)

        train_features =tf.fit_transform(processed_data)


        TF_save = dump(tf, "tfidf1_CLASS.pkl")

        #print('============the tokenizing is done here=========')
        return train_features,TF_save
