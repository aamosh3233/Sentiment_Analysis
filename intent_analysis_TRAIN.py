import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics.pairwise import cosine_similarity
from joblib import dump, load
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import  MultinomialNB

#### USING THESE MODELS

from sklearn.ensemble import RandomForestClassifier
from intent_analysis_FILTER import Filter
from sklearn.preprocessing import OneHotEncoder
df = pd.read_csv('training_intent_datasets.csv')


class Train:
    def __init__(self):
        df = pd.read_csv('training_intent_datasets.csv')
        preprocessor = Filter()
        le = LabelEncoder()
        self.le=le
        self.df=df
        self.preprocessor=preprocessor


    def featuring(self):
        # features with df['review']
        # print('it works up to here ==============================')
        self.df['sentence'] = self.df['sentence'].map(lambda x:self.preprocessor.message_process(x))
        y=self.le.fit(self.df['class'])

        y=y.transform(self.df['class'])

        # Training tf Vectorizer
        tf = TfidfVectorizer(ngram_range=(1, 2))
        # print('the tfs are=e=========',tf)



        train_features = tf.fit_transform(self.df['sentence'])

        train_Save=dump(train_features,'train.txt')



        TF_save = dump(tf, "tfidf1_INTENT.pkl")

        clf=MultinomialNB(alpha=0.2)
        clf.fit(train_features,y)



        clf_Save = dump(clf, "INTENT.pkl")

        print('train feautres=-===',train_features[0])
        return train_features,y,clf

        # print('============the tokenizing is done here=========')
    def changing_new_word_to_num(self,word):
        word = self.preprocessor.message_process(word)
        return word

train=Train()
train.featuring()