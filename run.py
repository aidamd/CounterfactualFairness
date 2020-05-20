from utils import *
import argparse
import pandas as pd
import json
from collections import Counter
from Counterfactual import *

def initialize_model(train_df, params):
    train_df = preprocess(train_df)

    train_text = train_df["text"].values.tolist()
    vocab = learn_vocab(train_text, params["vocab_size"])

    train_tokens = tokens_to_ids(train_text, vocab)

    hate_w = [1 - (Counter(train_df["hate"])[i] / len(train_df["hate"])) for i in [0, 1]]

    model = Counterfactual(params, vocab, hate_w)
    batches = get_batches(train_tokens,
                          model.batch_size,
                          vocab.index("<pad>"),
                          train_df["hate"].tolist())
    model.train(batches)
    return model, vocab

def test_model(test_df, vocab, model):
    test_df = preprocess(test_df)
    test_text = test_df["text"].values.tolist()

    test_tokens = tokens_to_ids(test_text, vocab)
    batches = get_batches(test_tokens,
                          params["batch_size"],
                          vocab.index("<pad>"))

    test_predictions = model.predict_hate(batches)
    prediction_results(test_df, test_predictions)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="Path to data; includes text, hate and offensive columns")
    parser.add_argument("--params", help="Parameter files. should be a json file")
    parser.add_argument("--test",)

    args = parser.parse_args()

    data = pd.read_csv(args.data)
    test = pd.read_csv(args.test)
    try:
        params = json.load(open(args.params, 'r'))
    except Exception:
        print("Wrong params file")
        exit(1)

    model, vocab, SGT = initialize_model(data, params)
    test_model(test, vocab, model, SGT)

