import re
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
nltk.download("stopwords")
nltk.download("punkt")
stop=set(stopwords.words('english'))


def fill_na(df):
    df['keyword'].fillna('', inplace=True)
    df['location'].fillna('', inplace=True)

    return df


def replace_locations(df):
    df['location'].replace({'United States':'USA',
                            'New York':'USA',
                            "London":'UK',
                            "Los Angeles, CA":'USA',
                            "Washington, D.C.":'USA',
                            "California":'USA',
                                "Chicago, IL":'USA',
                                "Chicago":'USA',
                            "New York, NY":'USA',
                            "California, USA":'USA',
                            "FLorida":'USA',
                            "Nigeria":'Africa',
                            "Kenya":'Africa',
                            "Everywhere":'Worldwide',
                            "San Francisco":'USA',
                            "Florida":'USA',
                            "United Kingdom":'UK',
                            "Los Angeles":'USA',
                            "Toronto":'Canada',
                            "San Francisco, CA":'USA',
                            "NYC":'USA',
                            "Seattle":'USA',
                            "Earth":'Worldwide',
                            "Ireland":'UK',
                            "London, England":'UK',
                            "New York City":'USA',
                            "Texas":'USA',
                            "London, UK":'UK',
                            "Atlanta, GA":'USA',
                            "Mumbai":"India"},
                            inplace=True
                            )


def drop_not_important_features(df):
    df.drop(['keyword','id','location','len'],axis=1,inplace=True)

    return df


def preprocess(text,stem=False):
    text = text.lower() # lowercase

    text = re.sub(r'[!]+', '!', text)
    text = re.sub(r'[?]+', '?', text)
    text = re.sub(r'[.]+', '.', text)
    text = re.sub(r"'", "", text)
    text = re.sub('\s+', ' ', text).strip()  # Remove and double spaces
    text = re.sub(r'&amp;?', r'and', text)  # replace & -> and
    text = re.sub(r"https?:\/\/t.co\/[A-Za-z0-9]+", "", text)  # Remove URLs
    text = re.sub(r'[:"$%&\*+,-/:;<=>@\\^_`{|}~]+', '', text)  # remove some puncts (except . ! # ?)
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'EMOJI', text)

    tokens = []
    for token in text.split():
        if token not in stop_words:
            if stem:
                tokens.append(SnowballStemmer.stem(token))
            else:
                tokens.append(token)

    return " ".join(tokens)


def apply_preprocess(df):
    stop_words = stopwords.words('english')
    stemmer = SnowballStemmer('english')
    lemmatizer = WordNetLemmatizer()

    df['text'] = df['text'].apply(lambda x: preprocess(x))

    return df


def correct_spell(text):
    spell = SpellChecker()

    corrected_text = []
    misspelled_words = spell.unknown(text.split())

    for word in text.split():
        if word in misspelled_words:
            corrected_text.append(spell.correction(word))
        else:
            corrected_text.append(word)

    return " ".join(corrected_text)


def apply_correct_check(df):
    df['text'] = df['text'].apply(lambda x:correct_spell(x))


def load_data_manipulation(df):
    df = fill_na(df)
    df = replace_locations(df)
    df = apply_preprocess(df)
    df = apply_correct_check(df)

    return df