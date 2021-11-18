import datetime as dt
import re

import pandas as pd
import streamlit as st
import time
import joblib

# Importing the StringIO module.
from io import StringIO

from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer

nltk.download('punkt')
nltk.download('stopwords')

stemmer = SnowballStemmer("german")
stop_words = set(stopwords.words("german"))


def clean_text(text, for_embedding=False):
    """
        - remove any html tags (< /br> often found)
        - Keep only ASCII + European Chars and whitespace, no digits
        - remove single letter chars
        - convert all whitespaces (tabs etc.) to single wspace
        if not for embedding (but e.g. tdf-idf):
        - all lowercase
        - remove stopwords, punctuation and stemm
    """
    RE_WSPACE = re.compile(r"\s+", re.IGNORECASE)
    RE_TAGS = re.compile(r"<[^>]+>")
    RE_ASCII = re.compile(r"[^A-Za-zÀ-ž ]", re.IGNORECASE)
    RE_SINGLECHAR = re.compile(r"\b[A-Za-zÀ-ž]\b", re.IGNORECASE)
    if for_embedding:
        # Keep punctuation
        RE_ASCII = re.compile(r"[^A-Za-zÀ-ž,.!? ]", re.IGNORECASE)
        RE_SINGLECHAR = re.compile(r"\b[A-Za-zÀ-ž,.!?]\b", re.IGNORECASE)

    text = re.sub(RE_TAGS, " ", text)
    text = re.sub(RE_ASCII, " ", text)
    text = re.sub(RE_SINGLECHAR, " ", text)
    text = re.sub(RE_WSPACE, " ", text)

    ## tokenize words
    word_tokens = word_tokenize(text)
    ## lower words
    words_tokens_lower = [word.lower() for word in word_tokens]

    
    if for_embedding:
        # no stemming, lowering and punctuation / stop words removal
        words_filtered = word_tokens
    else:
        words_filtered = [
            stemmer.stem(word) for word in words_tokens_lower if word not in stop_words
        ]

    text_clean = " ".join(words_filtered)
    return text_clean

class_mapping = {0: 'Etat',
                1: 'Inland',
                2: 'International',
                3: 'Kultur',
                4: 'Panorama',
                5: 'Sport',
                6: 'Web',
                7: 'Wirtschaft',
                8: 'Wissenschaft'}

# Set page title
st.title('10kGNAD Classification - Streamlit APP')

# Load classification model
with st.spinner('Loading classification model...'):
    # classifier = TextClassifier.load('models/best-model.pt')
    pipe = joblib.load('models/lsvc_tfidf.pkl')


st.subheader('German Text for classification')

text_input = st.text_input('Texts:')
if text_input != '':
    # Pre-process tweet
    # Make predictions
    with st.spinner('Predicting...'):
        cleaned_text = clean_text(text_input)
        pred = pipe.predict([text_input])

        st.write('Prediction:')
        st.write('Text classified as ' + class_mapping[pred[0]])




uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:


    # To convert to a string based IO:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))

    # To read file as string:
    string_data = stringio.read()

    with st.spinner('Predicting...'):
        cleaned_text = clean_text(string_data)
        pred = pipe.predict([cleaned_text])

        st.write('Prediction:')
        st.write('Text classified as ' + class_mapping[pred[0]])
