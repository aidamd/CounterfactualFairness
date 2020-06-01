import pandas as pd
import os
import random
import argparse
import math
import statistics

def clean(val):
    vec = [float(x) for x in val.replace("[", "").replace("]", "").rstrip().lstrip().split()]
    sum_exp = sum(math.exp(x) for x in vec)
    vex = [100 * math.exp(x) / sum_exp for x in vec]
    return  vex


def fair(path):
    x = ["stereo", "bias"]

    for file in x:
        s = pd.read_csv(os.path.join(path, file + "_context_predict.csv"))
        diffs = list()
        drop = list()
        for i, row in s.iterrows():
            if "5000" in row["text"]:
                drop.append(i)
        s = s.drop(drop)

        for name, group in s.groupby(["group"]):

            logits = [clean(str(row["logits"]))[1] for i, row in group.iterrows()]
            if file == "stereo":
                try:
                    main_logit = [clean(str(row["logits"]))[1] for i, row in group.iterrows() if row["orig_sgt"] == True][0]
                    logits.remove(main_logit)
                    diffs.append(statistics.variance(logits, main_logit))
                except Exception:
                    continue
            else:
                #main_logit = logits[random.randrange(len(logits))]
                #logits.remove(main_logit)
                diffs.append(statistics.variance(logits))

            #logits = [abs(log - main_logit) for log in logits]
            #diffs.append(sum(logits) / len(logits))

        print(file, sum(diffs) / len(diffs))

def tp(path):
    test = pd.read_csv(os.path.join(path, "gab_test_predict.csv"))
    tp, tn = dict(), dict()
    n, p = dict(), dict()
    for i, row in test.iterrows():
        if row["single_sgt"] != "":
            p[row["single_sgt"]] = p.get(row["single_sgt"], 0) + row["predict"]
            n[row["single_sgt"]] = n.get(row["single_sgt"], 0) + 1 - row["predict"]
            if row["predict"] + row["hate"] == 2:
                tp[row["single_sgt"]] = tp.get(row["single_sgt"], 0)  + 1
            if row["predict"] + row["hate"] == 0:
                tn[row["single_sgt"]] = tn.get(row["single_sgt"], 0) + 1
    tp_rate = [100 * tp[x] / p[x] for x in p.keys()]
    tn_rate = [100 * tn[x] / n[x] for x in n.keys()]

    print("true positive average", sum(tp_rate) / len(tp_rate), "variation", statistics.variance(tp_rate))
    print("true negative average", sum(tn_rate) / len(tn_rate), "variation", statistics.variance(tn_rate))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Path to data; includes text, hate and offensive columns")

    args = parser.parse_args()
    fair("saved_model/" + args.model)
    tp("saved_model/" + args.model)