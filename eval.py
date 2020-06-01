import pandas as pd
import json
import re
from Counterfactual import *
import argparse

def synth():
    df = pd.read_csv("Data/bias_77k.csv")
    new_df = {"group": list(),
              "text": list()}
    for name, group in df.groupby(["Template"]):
        if name == "verb_adj":
            for i, row in group.iterrows():
                new_df["group"].append(row.Text.split()[0])
                new_df["text"].append(row.Text)
        if name == "being_adj":
            for i, row in group.iterrows():
                new_df["group"].append(row.Text.split()[-1][:-1])
                new_df["text"].append(row.Text)
        if name == "you_are_adj":
            for i, row in group.iterrows():
                new_df["group"].append(row.Text.split()[3])
                new_df["text"].append(row.Text)
    pd.DataFrame.from_dict(new_df).to_csv("Data/bias_context.csv")


def stereo():
    df = json.load(open("Data/stereotype.json", "r"))

    new_df = {"sgt": list(),
              "text": list()}
    sgts = [sgt.replace("\n", "") for sgt in open("Data/extended_SGT.txt", "r").readlines()]

    for data in df["data"]["intersentence"]:
        sentence = data["context"]
        for sent in data["sentences"]:
            if sent["gold_label"] == "stereotype" and data["target"].lower() in sgts:
                stereotype = sentence + " " + sent["sentence"]
                new_df["text"].append(stereotype)
                new_df["sgt"].append(data["target"].lower())

    for data in df["data"]["intrasentence"]:
        for sent in data["sentences"]:
            if sent["gold_label"] == "stereotype" and data["target"].lower() in sgts:
                new_df["text"].append(sent["sentence"])
                new_df["sgt"].append(data["target"].lower())
                break

    counter_df = {"text": list(),
                  "group": list(),
                  "orig_sgt": list()}

    for i in range(len(new_df["text"])):
        for sgt in sgts:
            counter_df["text"].append(new_df["text"][i].lower().replace(new_df["sgt"][i], sgt))
            counter_df["group"].append(i)
            counter_df["orig_sgt"].append(new_df["sgt"][i] == sgt)

    pd.DataFrame.from_dict(counter_df).to_csv("Data/stereo_context.csv")

def eval_logits(test_path):
    df = pd.read_csv(test_path)


def eval_test(test_path, param_file):
    params = json.load(open(param_file, 'r'))
    model = Counterfactual(params)
    test = pd.read_csv(test_path)

    test = model.test_model(test)
    test.to_csv(os.path.join("saved_model/", params["type"], test_path.split("/")[-1].split(".")[0] + "_predict.csv")
                , index=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="Path to data; includes text, hate and offensive columns")
    parser.add_argument("--params")

    args = parser.parse_args()
    eval_test(args.data, args.params)


