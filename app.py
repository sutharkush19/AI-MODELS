import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from joblib import load



ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)


with open('vectorized.pkl', 'rb') as f:
    tfidf = pickle.load(f)

model = load('model.joblib')


st.title('Sms Spam Classifier')

input_sms = st.text_input('Enter the Message')

if st.button('Predict'):

     #preprocess

     transformed_sms = transform_text(input_sms)

     #vectorizing

     vector_input = tfidf.transform([transformed_sms])

     #predict

     result = model.predict(vector_input)[0]

     #display

     if result == 1:
         st.header('Sms Spam')
     else:
         st.header('Sms Not Spam')
