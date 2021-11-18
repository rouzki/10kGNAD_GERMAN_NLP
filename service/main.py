from fastapi import FastAPI
import os
import numpy as np
import uvicorn
from pydantic import BaseModel
import pandas as pd
import joblib
import re

from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer

nltk.download('punkt')
nltk.download('stopwords')

stemmer = SnowballStemmer("german")
stop_words = set(stopwords.words("german"))


PKL_FILENAME = "lsvc_tfidf.pkl"
MODELS_PATH = "models/"
MODEL_FILE_PATH = os.path.join(MODELS_PATH,PKL_FILENAME)


pipeline = joblib.load(MODEL_FILE_PATH)


class_mapping = {0: 'Etat',
                1: 'Inland',
                2: 'International',
                3: 'Kultur',
                4: 'Panorama',
                5: 'Sport',
                6: 'Web',
                7: 'Wirtschaft',
                8: 'Wissenschaft'}

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

app = FastAPI(title="10kGNAD Classification - Streamlit APP", description="API to predict class of german text")

class Data(BaseModel):
    text:str

@app.get("/")
def read_root():
    return {"message": "Welcome from the API"}


@app.post("/predict")
def predict(data:Data):


    cleaned_text = clean_text(data.text)
    prediction = pipeline.predict([cleaned_text])
    print(data.text)

    return {
        
        "Text" : data.text,
        "prediction": class_mapping[prediction[0]],

    }


if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)