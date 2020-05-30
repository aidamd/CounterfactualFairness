import argparse
import pandas as pd
import json
from Counterfactual import *
from sklearn.model_selection import StratifiedKFold

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="Path to data; includes text, hate and offensive columns")
    parser.add_argument("--counter", help="Path to counterfactuals")
    parser.add_argument("--params", help="Parameter files. should be a json file")

    args = parser.parse_args()
    data = pd.read_csv(args.data)
    counter = pd.read_csv(args.counter)
    try:
        params = json.load(open(args.params, 'r'))
    except Exception:
        print("Wrong params file")
        exit(1)

    test = pd.read_csv(args.test)
    model = Counterfactual(params, data, counter)
    model.CV()

