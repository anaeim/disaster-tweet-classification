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
    """fills na in pandas.DataFrame with an empty string

    Parameters
    ----------
    df : pandas.DataFrame
        tweet classification dataset

    Returns
    -------
    pandas.DataFrame
        tweet classification dataset after pre-processing
    """

    df['keyword'].fillna('', inplace=True)
    df['location'].fillna('', inplace=True)

    return df


def replace_locations(df):
    """replaces city locations with the country that they located in, replaces country names with their abbreviated names

    Parameters
    ----------
    df : pandas.DataFrame
        tweet classification dataset

    Returns
    -------
    pandas.DataFrame
        tweet classification dataset after pre-processing
    """

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

    return df


def drop_not_important_features(df):
    """drops features (pandas.DataFrame columns) that are less important for disaster tweet classification task

    Parameters
    ----------
    df : pandas.DataFrame
        tweet classification dataset

    Returns
    -------
    pandas.DataFrame
        tweet classification dataset after pre-processing
    """

    df.drop(['keyword','id','location','len'],axis=1,inplace=True)
    return df


def preprocess(text,stem=False):
    """applies pre-processing steps using regex functions, removes stop words for the text of one tweet

    Parameters
    ----------
    text : str
        the text of one tweet

    Returns
    -------
    str
        the text of one tweet after pre-processing
    """

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
    """applies pre-processing steps using regex functions, removes stop words for the text of all the tweets in the dataset

    Parameters
    ----------
    df : pandas.DataFrame
        tweet classification dataset

    Returns
    -------
    pandas.DataFrame
        tweet classification dataset after pre-processing
    """

    stop_words = stopwords.words('english')
    stemmer = SnowballStemmer('english')
    lemmatizer = WordNetLemmatizer()

    df['text'] = df['text'].apply(lambda x: preprocess(x))

    return df


def correct_spell(text):
    """applies spell correction for misspelled words in a the text of one tweet

    Parameters
    ----------
    text : str
        the text of one tweet

    Returns
    -------
    str
        the text of one tweet after pre-processing
    """

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
    """applies spell correction for misspelled words in a the text of all the tweets in the dataset

    Parameters
    ----------
    df : pandas.DataFrame
        tweet classification dataset

    Returns
    -------
    pandas.DataFrame
        tweet classification dataset after pre-processing
    """

    df['text'] = df['text'].apply(lambda x:correct_spell(x))
    return df


def load_data_manipulation(df):
    """applies all the data manipulation functions on the dataset

    Parameters
    ----------
    df : pandas.DataFrame
        tweet classification dataset

    Returns
    -------
    pandas.DataFrame
        tweet classification dataset after pre-processing
    """

    df = fill_na(df)
    df = replace_locations(df)
    df = apply_preprocess(df)
    df = apply_correct_check(df)

    return df