import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import dump, load
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LinearRegression
from sklearn import svm
from DATA_FILTER import Filter
from flask import Flask,request, jsonify




import pickle
import sys

preprocessor=Filter()


class Test:
    def __init__(self):
        print('the test works')



    def words(self,arg):

        arg = ''.join(arg)
        print('arg======',arg)
        new_text =preprocessor.message_process(arg)
        print('new text==========',new_text)

        # loading old vectorizer with dump and load
        TF1 =load('tfidf1_CLASS.pkl')
        print('TF1===',TF1)

        ### adding old vocab with the new ones
        TF2 = TfidfVectorizer(vocabulary=TF1.vocabulary_)
        print('TF2================',TF2)
        # print('TF_NEW ko vectorizer pd series===',TF2)
        X = pd.Series(new_text)
        print('X============',X)
        # the feature
        features = TF2.fit_transform(X)

        print('features======',features)

        model = load('NOMINAL_CLASS.joblib')
        pred = model.predict(features)
        # print('the model is ==',pred)

        if pred == 0:
            return 'the prediction is bad'
        else:
            return 'the prediction is good'

