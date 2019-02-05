import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import dump, load
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LinearRegression
from sklearn import svm
from DATA_TEST import Test
from flask import Flask,request, jsonify




res=Test()
#res.words('not bad')

app = Flask(__name__)


x={'review':'good'}
y={'another review':'bad'}
df=pd.read_csv('data.tsv',sep='\t',names=['review','sentiment'])


@app.route('/',methods=['POST'])
def side():
    return jsonify({"THE MOVIE IS ": x['review']})

@app.route('/pra',methods=['POST'])
def pra():
    y = request.json['review']
    X=res.words(y)
    return jsonify({'the outcome is ':X})

if __name__ == '__main__':
    app.run(debug=True)