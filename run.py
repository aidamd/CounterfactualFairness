import argparse
import pandas as pd
import json
from Counterfactual import *
from sklearn.model_selection import StratifiedKFold

def initialize_model(train_df, counter, params):
    train_df = preprocess(train_df)
    counter_df = preprocess(counter)

    train_text = train_df["text"].values.tolist()
    train_ids = train_df["Tweet ID"].values.tolist()
    train_labels = train_df["hate"].tolist()
    vocab = learn_vocab(train_text, params["vocab_size"])

    train_tokens = tokens_to_ids(train_text, vocab)
    counterfactuals = dict()
    for name, group in counter.groupby(["Tweet ID"]):
        counterfactuals[name] = tokens_to_ids(group["text"].values.tolist(), vocab)

    hate_w = [1 - (Counter(train_df["hate"])[i] / len(train_df["hate"])) for i in [0, 1]]

    model = Counterfactual(params, vocab, hate_w)

    kfold = StratifiedKFold(n_splits=5)

    results = dict()
    i = 1
    for t_idx, v_idx in kfold.split(np.arange(len(train_tokens)),
                                          train_labels):
        print("CV:", i)
        i += 1
        t_tokens, v_tokens = [train_tokens[t] for t in t_idx], [train_tokens[t] for t in v_idx]
        t_index, v_index = [train_ids[t] for t in t_idx], [train_ids[t] for t in v_idx]
        t_labels, v_labels = [train_labels[t] for t in t_idx], [train_labels[t] for t in v_idx]
        train_batches = get_batches(t_tokens,
                              t_index,
                              model.batch_size,
                              vocab.index("<pad>"),
                              hate=t_labels,
                              counter=counterfactuals)
        val_batches = get_batches(v_tokens,
                              v_index,
                              model.batch_size,
                              vocab.index("<pad>"),
                              hate=v_labels,
                              counter=counterfactuals)
        model.build()
        model.train(train_batches, val_batches)
        v_predictions = model.predict(val_batches)
        res = prediction_results(v_labels, v_predictions)
        for m in res:
            results.get(m, list()).append(res[m])
    for m in results:
        print(m, ":", sum(results[m]) / len(results[m]))
    return model, vocab

def test_model(test_df, vocab, model):
    test_df = preprocess(test_df)
    test_text = test_df["text"].values.tolist()
    test_idx = test_df["Tweet ID"].values.tolist()

    test_tokens = tokens_to_ids(test_text, vocab)
    batches = get_batches(test_tokens,
                          test_idx,
                          params["batch_size"],
                          vocab.index("<pad>"))

    test_predictions = model.predict(batches)
    _ = prediction_results(test_df["hate"].values.tolist(),
                           test_predictions)


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

