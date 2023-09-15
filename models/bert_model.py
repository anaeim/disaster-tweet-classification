import re
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from nltk.util import ngrams
from nltk.tokenize import word_tokenize

import transformers
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer
from transformers import TFBertModel

import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Conv1D, Bidirectional, LSTM, Dense, Input, Dropout
from tensorflow.keras.layers import SpatialDropout1D, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.utils.vis_utils import plot_model


class TweetClassifier:
    """text classification model for disaster tweets

    ...

    Attributes
    ----------
    x_train : pandas.DataFrame
        dataset input for training the model
    x_valid : pandas.DataFrame
        dataset input for model validation
    y_train : pandas.DataFrame
        dataset target for training the model
    y_valid : pandas.DataFrame
        dataset target for model validation
    args : namedtuple
        the arguments pre-defined by the user and imported from config.py

    Methods
    -------
    tokenization_padding()
        applies necessary tokenization and padding for the LLM to reach the specified length
    bert_encoder(texts, maximum_length)
        encodes text based on pre-defined maximum_length and the encoder of the LLM
    load_lm()
        loads a LLM from transformers library based on the pre-defined LLM model
    create_model()
        creates a deep neural networks model using tensorflow library
    fit(x_train, y_train, args)
        fits the deep neural networks model on the x_train and y_train data considering the pre-defined args (arguments)
    """

    def __init__(self):
        self.x_train = None
        self.x_valid = None
        self.y_train = None
        self.y_valid = None
        self.args = None

    def tokenization_padding(self):
        """applies necessary tokenization and padding for the LLM to reach the specified length
        """

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(self.x_train)
        word_index = tokenizer.word_index
        vocab_size = len(word_index) + 1
        print('The Vocabulary Size is Equal to: {}'.format(vocab_size))

        self.x_train = pad_sequences(tokenizer.texts_to_sequences(self.x_train['text']), maxlen = 30)
        self.x_valid = pad_sequences(tokenizer.texts_to_sequences(self.x_valid['text']), maxlen = 30)

    @staticmethod
    def bert_encoder(texts, maximum_length):
        """encodes text based on pre-defined maximum_length and the encoder of the LLM

        Parameters
        ----------
        texts : pandas.DataFrame
            text column of the input data
        maximum_length : int
            maximum_length pre-defined by the user from config.py

        Returns
        -------
        ndarray,ndarray
            id of words,attention masks of words
        """

        tokenizer = BertTokenizer.from_pretrained(self.args.lm)

        input_ids = []
        attention_masks = []

        for text in texts:
            encoded = tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=maximum_length,
                pad_to_max_length=True,
                return_attention_mask=True,
            )
            input_ids.append(encoded['input_ids'])
            attention_masks.append(encoded['attention_mask'])

        return np.array(input_ids),np.array(attention_masks)

    def load_lm(self):
        """loads a LLM from transformers library based on the pre-defined LLM model
        """
        return TFBertModel.from_pretrained(self.args.lm)

    def create_model(self):
        """creates a deep neural networks model using tensorflow library
        """

        input_ids = tf.keras.Input(shape=(60,),dtype='int32')
        attention_masks = tf.keras.Input(shape=(60,),dtype='int32')

        input_ids, attention_masks = self.bert_encoder(self.x_train,60)
        lm_model = self.load_lm()

        output = lm_model([input_ids,attention_masks])
        output = output[1]
        output = tf.keras.layers.Dense(32,activation='relu')(output)
        output = tf.keras.layers.Dropout(0.2)(output)
        output = tf.keras.layers.Dense(1,activation='sigmoid')(output)

        model = tf.keras.models.Model(inputs = [input_ids,attention_masks],outputs = output)
        model.compile(Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

        return model

    def fit(self, x_train, y_train, args):
        """fits the deep neural networks model on the x_train and y_train data considering the pre-defined arguments

        Parameters
        ----------
        x_train : pandas.DataFrame
            dataset input for training the model
        y_train : pandas.DataFrame
            dataset target for training the model
        args : namedtuple
            the arguments defined by the user and import from config.py
        """

        self.x_train = x_train
        self.y_train = y_train
        self.args = args
        # self.train_test_split()
        self.tokenization_padding()

        self.model = self.create_model()
        input_ids, attention_masks = self.bert_encoder(self.x_train,60)

        self.model.fit(
        [input_ids, attention_masks],
        self.y_train,
        epochs=self.args.epochs,
        batch_size=self.args.batch_size
        )