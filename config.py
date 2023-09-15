import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    
    add_arg("--dataset-path", default="./data", help="directory that you stored the dataset", type=str)
    add_arg("--ml-model", default="bert_model", choices=["bert_model"], help="ML model for the prediction", type=str)
    add_arg("--lm", default="bert-large-uncased", choices=["bert-base-uncased", "bert-base-cased", "bert-large-cased", "bert-large-uncased"], help="llm model for the prediction", type=str)
    add_arg("--validation_split", default=0.2, help="proportion between train and valid datasets")
    add_arg("--epochs", default=3, type=int)
    add_arg("--batch_size", default=10, type=int)

    return parser.parse_args()

if __name__ == "__main__":
    print(parse_args())