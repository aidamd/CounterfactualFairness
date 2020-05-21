from utils import *
import argparse
import pandas as pd
import json
from collections import Counter
from Counterfactual import *

def initialize_model(train_df, counter, params):
    train_df = preprocess(train_df)
    counter_df = preprocess(counter)

    train_text = train_df["text"].values.tolist()
    train_idx = train_df["Tweet ID"].values.tolist()
    vocab = learn_vocab(train_text, params["vocab_size"])

    train_tokens = tokens_to_ids(train_text, vocab)
    counterfactuals = dict()
    for name, group in counter.groupby(["Tweet ID"]):
        counterfactuals[name] = tokens_to_ids(group["text"].values.tolist(), vocab)

    hate_w = [1 - (Counter(train_df["hate"])[i] / len(train_df["hate"])) for i in [0, 1]]

    model = Counterfactual(params, vocab, hate_w)
    batches = get_batches(train_tokens,
                          train_idx,
                          model.batch_size,
                          vocab.index("<pad>"),
                          train_df["hate"].tolist(),
                          counter=counterfactuals)
    model.build()
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
    parser.add_argument("--counter", help="Path to counterfactuals")
    parser.add_argument("--params", help="Parameter files. should be a json file")
    parser.add_argument("--test",)

    args = parser.parse_args()
    data = pd.read_csv(args.data)
    counter = pd.read_csv(args.counter)
    try:
        params = json.load(open(args.params, 'r'))
    except Exception:
        print("Wrong params file")
        exit(1)

    model, vocab= initialize_model(data, counter, params)
    test = pd.read_csv(args.test)
    test_model(test, vocab, model)

