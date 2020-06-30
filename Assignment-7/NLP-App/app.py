import streamlit as st
import pandas as pd
import numpy as np
from  Setup import System

v=System()

def EDA():
    st.sidebar.subheader('Data size')
    r,c=v.shape()
    if st.sidebar.selectbox('Number of: ',['Rows','Columns'])=='Rows':
        st.info('Number of Rows: '+str(r))
    else:
        st.info('Number of Columns: '+str(c))

    st.sidebar.subheader('Data Info')
    user_input = st.sidebar.text_input("Enter amount of rows to display (MAX: 5572)", 5)
    st.table(v.head(int(user_input)))

    st.sidebar.subheader('Spams or Hams')
    if st.sidebar.selectbox('Number of: ',['Spams','Hams'])=='Spams':
        st.info('Number of Spams: '+str(v.values('spam')))
        st.image('data/spam.png',use_column_width=True)
        st.pyplot()
    else:
        st.info('Number of Hams: '+str(v.values('ham')))
        st.image('data/ham.png',use_column_width=True)
        st.pyplot()

def Predictor():
    st.image('data/spam_ham.jpg')
    models=['Logistic Regression','Decission Tree','SVM']
    model=st.sidebar.selectbox("Choose Model to Use: ",models)
    if model==models[0]:
        st.subheader('Logistic Predictor')
        txt=st.text_input('Enter message here')
        st.info('Entered message is '+v.Logistic_pred(txt))

    if model==models[1]:
        st.subheader('Decission Tree Predictor')
        txt=st.text_input('Enter message here')
        st.info('Entered message is '+v.Decission_tree_pred(txt))

    if model==models[2]:
        st.subheader('SVM Predictor')
        txt=st.text_input('Enter message here')
        st.info('Entered message is '+v.SVM_pred(txt))

def main():
    st.title("Spam-Ham Detector")

    st.sidebar.warning("<> with :heart: for **Cipherschools**")
    st.sidebar.error("Checkout [Github](https://github.com/rak3n/CIPHERSCHOOLS_Assignments/Assignment-7/NLP-App) for **Source**")

    st.sidebar.header('Choose a option option')

    type=st.sidebar.selectbox(
    'App mode',
    ['Visualizer','Detector']
    )

    if type=='Visualizer':
        EDA()
    else:
        Predictor()


if __name__=="__main__":
    main()
