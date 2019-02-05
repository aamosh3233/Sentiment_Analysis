import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import dump, load
import pandas as pd
from intent_TEST import Test
from intent_analysis_TRAIN import Train

from flask import Flask,request, jsonify

app = Flask(__name__)

word1={'word1=':'write word here'}
word2={'word2=':'write another word here'}

df = pd.read_csv('training_intent_datasets.csv')
class API:
    def __init__(self):

        self.df = pd.read_csv('training_intent_datasets.csv')

    def result(self,word1,word2):
        Test.test(word1, word2)
        cosine_score=Test.cosine_score(word1,word2)
        jaccard=Test.get_jaccard_sim(word1,word2)
        return cosine_score,jaccard


api=API()
test=Test()
train=Train()

@app.route('/classify',methods=['POST'])
def side():
    y1 = request.json['word=']
    Z1=test.splitting(y1)
    predict=test.prediction(y1)
    return jsonify({"THE word is ":predict})
###yo line na chune ######3



#### JACCARD KO LAGI ########

@app.route('/jaccard',methods=['POST'])
def jac():
    test = Test()

    y1= request.json['word=']
    X=test.splitting(y1)
    #given word and training dataset


    jacard = test.get_jaccard_sim(X,)

    return jsonify({'the jaccard is ':jacard})
#########################333

@app.route('/cosine',methods=['POST'])
def cos():
    test = Test()
    y1=request.json['word=']

    z=test.splitting()
    cosine,jacard=api.result(x,z)

    return jsonify({'the cosine is ':cosine})

if __name__ == '__main__':
    app.run(debug=True)