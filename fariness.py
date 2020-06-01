import pandas as pd
import os
import random
import argparse

def clean(val):
    return [float(x) for x in val.replace("[", "").replace("]", "").rstrip().lstrip().split()]


def fair(path):
    x = ["stereo", "bias"]

    for file in x:
        s = pd.read_csv(os.path.join(path, file + "_context_predict.csv"))
        diffs = list()

        for name, group in s.groupby(["group"]):

            logits = [clean(str(row["logits"]))[1] for i, row in group.iterrows()]
            if file == "stereo":
                main_logit = [clean(str(row["logits"]))[1] for i, row in group.iterrows() if row["orig_sgt"] == True][0]
            else:
                main_logit = logits[random.randrange(len(logits))]
            logits.remove(main_logit)
            logits = [abs(log - main_logit) for log in logits]
            diffs.append(sum(logits) / len(logits))

        print(file, sum(diffs) / len(diffs))

def tp(path):
    test = pd.read_csv(os.path.join(path, "gab_test_predict.csv"))




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Path to data; includes text, hate and offensive columns")

    args = parser.parse_args()
    fair("saved_model/" + args.model)
    tp("saved_model/" + args.model)