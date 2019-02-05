import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump, load

#### USING THESE MODELS
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LinearRegression
from sklearn import svm
from DATA_FILTER import Filter

preprocessor=Filter()
df = pd.read_csv('data.tsv', sep='\t', names=['review', 'sentiment'])
print(df)
class Train:

    def __init__(self,df):

        self.df=df


    def split(self,df):
        #getting the dataset ready

        processed_data=preprocessor.featuring(df)
        train_features,TF_save=preprocessor.tokenizing(processed_data)
        xtrain, xtest, ytrain, ytest = train_test_split(train_features, df['sentiment'], test_size=0.2)
        self.xtrain=xtrain
        self.xtest=xtest
        self.ytrain=ytrain
        self.ytest=ytest

        '''
        print('xtrain=====  \n',xtrain)
        print('xtest=====  \n', xtest)

        print('ytrain=====  \n', ytrain)
        print('ytest=====  \n', ytest)
'''
        print('============the spliting is done here=========')

    def random_forest(self):
        clf1 = RandomForestClassifier(n_estimators=30)
        clf1.fit(self.xtrain, self.ytrain)
        pred = clf1.predict(self.xtest)
        X = accuracy_score(pred, self.ytest)
        print('random forest acc==', X)
        randomforest = dump(clf1, 'Randomforest_CLASS.joblib')
        print('============the random forest is done here=========')

        return randomforest

    def logistics(self):
        # logistics
        clf2 = LogisticRegression()
        clf2.fit(self.xtrain, self.ytrain)
        pred = clf2.predict(self.xtest)

        X = accuracy_score(pred, self.ytest)
        #print('logistics acc==', X)
        logistics = dump(clf2, 'LOGISTICS_CLASS.joblib')
        #print('============the training is done here=========')

        return logistics

    def  linear(self):
        linar = MultinomialNB()
        linar.fit(self.xtrain, self.ytrain)
        pred = linar.predict(self.xtest)
        X = accuracy_score(pred, self.ytest)
        print('linear acc==', X)
        linear = dump(linar, 'LINEAR_CLASS.joblib')
        return linear
    def MultinomialNB(self):
        clf3 = MultinomialNB()
        clf3.fit(self.xtrain, self.ytrain)
        pred = clf3.predict(self.xtest)
        X = accuracy_score(pred, self.ytest)
        print('nominal acc==', X)
        randomforest = dump(clf3, 'NOMINAL_CLASS.joblib')
        return  randomforest

train=Train(df); train.split(df)

