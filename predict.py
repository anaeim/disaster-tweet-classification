from config import parse_args
from dataloader import data_loader
from datamanipulation import data_manipulation
import models


def main():
    args = parse_args()
    df = data_loader.load()
    df = data_manipulation.load_data_manipulation(df)
    model = {
        'bert_model': models.bert_model
    }[args.ml-model]

    model = model.TweetClassifier(df, args)
    model.fit()

if __name__ == "__main__":
    main()