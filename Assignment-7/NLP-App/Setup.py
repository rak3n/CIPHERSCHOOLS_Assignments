import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import seaborn as sb
from wordcloud import WordCloud,STOPWORDS
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import pickle
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from tqdm import tqdm
tqdm.pandas()

stemmer = PorterStemmer()

def preprocess(raw_tweet,setting):
    #regex to remove all characters except alphabets
    clean = re.sub("[^a-zA-Z]", " ",raw_tweet)
    #regex to remove all repeating sequence of letters
    clean= re.sub(r'([a-z])\1+', r'\1',clean)
    # change sentence to lower case
    clean = clean.lower()
    # tokenize into words
    words = clean.split()
    # remove stop words
    words = [w for w in words if not w in stopwords.words("english")]
    if setting=='stem':
        #Stemming
        words = [stemmer.stem(word) for word in words]

    return(" ".join(words))

class System():

    def __init__(self):
        self.df=pd.read_csv("data/SMSSpamCollection",sep="\t", names=["type","text"])


    def shape(self):
        row,column=self.df.shape
        return row,column

    def head(self,n=5):
        return self.df.head(n)

    def values(self,type):
        sb.countplot(x='type', data=self.df)
        plt.plot()
        return len(self.df.loc[self.df["type"]==type])

    def wordcloud(self):
        ham_df=self.df.loc[self.df.type=="ham",'text']
        spam_df=self.df.loc[self.df.type=="spam",'text']
        HAM_words=' '.join(ham_df)
        SPAM_words=' '.join(spam_df)

        wordcloud_ham = WordCloud(stopwords=STOPWORDS,background_color='black',width=1000,height=500).generate(HAM_words)
        wordcloud_spam=WordCloud(stopwords=STOPWORDS,background_color='black',width=1000,height=500).generate(SPAM_words)

        wordcloud_ham.to_file('ham.png')
        wordcloud_spam.to_file('spam.png')

    def Logistic_pred(self,message):
        v=pickle.load(open('pickle/vectorizer.pkl','rb'))
        model=pickle.load(open('pickle/Logistic.pkl','rb'))
        if model.predict(v.transform([preprocess(message,'stem')]).toarray()):
            return 'Spam'
        else:
            return 'Ham'

    def Decission_tree_pred(self,message):
        v=pickle.load(open('pickle/vectorizer.pkl','rb'))
        model=pickle.load(open('pickle/Decission.pkl','rb'))
        if model.predict(v.transform([preprocess(message,'stem')]).toarray()):
            return 'Spam'
        else:
            return 'Ham'

    def SVM_pred(self,message):
        v=pickle.load(open('pickle/vectorizer.pkl','rb'))
        model=pickle.load(open('pickle/SVM.pkl','rb'))
        if model.predict(v.transform([preprocess(message,'stem')]).toarray()):
            return 'Spam'
        else:
            return 'Ham'

    def DataPrep_and_Model(self):
        self.df['spam']=self.df['type'].progress_apply(lambda x:1 if x=='spam' else 0)
        self.df['clean_text_stem']=self.df['text'].progress_apply(lambda x: preprocess(x,'stem'))
        X=self.df['clean_text_stem']
        Y=self.df.spam

        X_train,x_test,Y_train,y_test=train_test_split(X,Y,train_size=0.7,random_state=60)
        v = CountVectorizer(analyzer = "word")
        X_train= v.fit_transform(list(X_train))
        x_test=v.transform(list(x_test))

        #Modeling
        log_model=LogisticRegression(verbose=True)
        log_model.fit(X_train.toarray(),Y_train)

        dt_model = DecisionTreeClassifier()
        dt_model.fit(X_train.toarray(), Y_train)

        svc_model=SVC()
        svc_model.fit(X_train.toarray(),Y_train)


        #Saving the Model and CountVectorizer
        pickle.dump(v,open('pickle/vectorizer.pkl','wb'))
        pickle.dump(log_model,open('pickle/Logistic.pkl','wb'))
        pickle.dump(dt_model,open('pickle/Decission.pkl','wb'))
        pickle.dump(svc_model,open('pickle/SVM.pkl','wb'))
        print("Completed")


"""
v=Visual()
v.wordcloud()
"""

"""
print("Starting process")
v=System()
v.DataPrep_and_Model()
"""
