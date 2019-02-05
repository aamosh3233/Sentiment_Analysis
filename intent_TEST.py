import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import dump, load
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LinearRegression
from sklearn import svm
from intent_analysis_FILTER import Filter
from intent_analysis_TRAIN import Train
from flask import Flask,request, jsonify
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle
import sys

from sklearn.metrics import jaccard_similarity_score


class Test:
    def __init__(self):
        self.preprocessor = Filter()
        self.train=Train()

    def test(self,word1,word2):

        train=load('train.txt')
        tf = load("tfidf1_INTENT.pkl")

        TF2 = TfidfVectorizer(vocabulary=tf.vocabulary_)

        # preparing for word 1
        X = pd.Series(word1)
        features1 = TF2.fit_transform(X)

        # preparing for word 2

        X = pd.Series(word2)
        features2 = TF2.fit_transform(X)

        return features1,features2


    def splitting(self,word1):
        train = load('train.txt')
        tf = load("tfidf1_INTENT.pkl")
        TF2 = TfidfVectorizer(vocabulary=tf.vocabulary_)

        # preparing for word 1

        X = pd.Series(word1)
        features1 = TF2.fit_transform(X)
        return features1


    def prediction(self,word1):

        x,y,clf=self.train.featuring()
        feature1=self.splitting(word1)
        prediction=clf.predict(feature1)
        if prediction == 0:
            return 'goodbye'
        elif prediction == 1:
            return 'greeting'
        else:
            return 'order'

    def get_jaccard_sim(self,str1,str2):

        a = set(str1.split())
        print('a==',a)
        b = set(str2.split())
        print('b==', b)
        c = a.intersection(b)
        print('c==', c)
        score= float(len(c)) / (len(a) + len(b) - len(c))
        return score

    def cosine_score(self,word1,word2):
        feature1,feature2=self.test(word1,word2)
        score = cosine_similarity(feature1,feature2)
        score=score.reshape(1,)
        print('the cosine score isss=-----------', score)
        return score

