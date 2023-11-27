from flask import Flask, render_template, flash, request, url_for, redirect, session
import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

import gensim
from gensim.utils import tokenize
def transform_text(text):
    text = text.lower()
    text = list(tokenize(text))

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('logistic_model.pkl','rb'))
app = Flask(__name__)
@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/detection', methods = ['POST'])
def spam_detection():
    
    if request.method=='POST':
        text = request.form['textInput']
        
        detect = ''
        transformed_sms = transform_text(text)
        # 2. vectorize
        vector_input = tfidf.transform([transformed_sms])
        # 3. predict
        result = model.predict(vector_input)[0]
        # 4. Display
        if result == 1:
            detect='Spam'
        else:
            detect='Not Spam'

       
    return render_template('index.html', text=text,detect=detect)


if __name__ == "__main__":
    app.run(debug=True)