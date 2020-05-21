import pandas as pd
from nltk import tokenize as nltk_token

df = pd.read_csv("Data/gab_test.csv")
sgts = [s.replace("\n", "") for s in open("Data/extended_SGT.txt").readlines()]

sample = {"text": list(),
          "Tweet ID": list(),
          "SGT": list(),
          "original_SGT": list(),
          "hate": list()}
for i, row in df.iterrows():
    for s in sgts:
        if s in nltk_token.WordPunctTokenizer().tokenize(row["text"].lower()):
            for sub in sgts:
                sample["text"].append(row["text"].lower().replace(s, sub))
                sample["Tweet ID"].append(row["Tweet ID"])
                sample["SGT"].append(sub)
                sample["original_SGT"].append(s)
                sample["hate"].append(row["hate"])
            continue

pd.DataFrame.from_dict(sample).to_csv("Data/counter.csv", index=False)