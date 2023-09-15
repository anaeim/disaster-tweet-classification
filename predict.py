from config import parse_args
from dataloader import data_loader
from datamanipulation import data_manipulation
import models


def main():
    args = parse_args()
    df = data_loader.load()
    df = data_manipulation.load_data_manipulation(df)
    train_data, valid_data = train_test_split(df, test_size=args.validation_split)
    x_train = train_data['text']
    x_valid = valid_data['text']
    y_train = train_data['target']
    y_valid = valid_data['target']

    Model = {
        'bert_model': models.bert_model.TweetClassifier
    }[args.ml-model]

    model = Model()
    model.fit(x_train, y_train, args)

if __name__ == "__main__":
    main()