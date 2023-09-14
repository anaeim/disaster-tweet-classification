import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg("--dataset-path", default="./data", help="directory that you stored the dataset", type=str)
    add_arg("--model", default="bert", choices=["bert"], help="ML model for the prediction", type=str)

    return parser.parse_args()

if __name__ == "__main__":
    print(parse_args())