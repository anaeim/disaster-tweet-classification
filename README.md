# LLM-based Disaster Tweet Classification


# Intro
Our model is a text classification solution based Large Language Models (LLMs) such as BERT. Given the text of a tweet, the model checks if the tweet is about a piece of real disaster news or a fake one. My solution leverages the powerful language understanding capability inherent in LLMs via fine-tuning. My model effectively encodes the tweets and utilizes a 4-layer fully-connected neural network as the classification head to predict the label. Additionally, I use other optimization techniques such as summarization, data manipulation, and pre-processing to improve the LLM-based model performance.


# Installation
To get started, you'll need Python and pip installed.

1. Clone the Git repository
```
git clone https://github.com/anaeim/disaster-tweet-classification.git
```

2. Navigate to the project directory
```
cd disaster-tweet-classification
```

3. Create a directory for data <br>
   The data is accessible on the [Kaggle website](https://www.kaggle.com/competitions/nlp-getting-started/data).
   
```
mkdir data
```

4. Install the requirements
```
pip install -r requirements.txt
```

# Training
```
python predict.py --dataset-path data \
    --ml-model bert_model \
    --lm bert-large-uncased \
    --validation_split 0.2 \
    --epochs 3 \
    --batch_size 10
```
The meaning of the flags:
* ``--dataset-path``: the directory that contains the dataset
* ``--ml-model``: the Machine Learning (ML) model that we use. Here I only include LLM-based tweet classification solution that is based on BERT model.
* ``--lm``: the language model. We now support ``bert-base-uncased``, ``bert-base-cased``, ``bert-large-uncased``, and ``roberta`` (``bert-large-uncased`` by default).
* ``--validation_split``, ``--epochs``, and ``--batch_size`` are the validation size, number of epochs and number of training examples in each iteration of the model, respectively.