import os
import sys
import pandas as pd
from tqdm import tqdm
from preprocessing import Preprocessing
import glob
from pathlib import Path

def counting_records():
    predict_dir = sys.argv[1]
    count = 0
    for filename in tqdm(os.listdir(predict_dir)):
        # if filename.endswith(".csv"):
        print("--------------------------------")
        print(os.path.join(predict_dir, filename))
        print("--------------------------------")
        # else:
        #     continue
        dataset = pd.read_csv(os.path.join(predict_dir, filename))
        print(dataset.shape[0])
        count += dataset.shape[0]

    print("Total number of records =", count)

def get_single_sgts():
    first = True
    for file in glob.glob("dump/Populated_SGT_chunk_*"):
        print("Reading", file)
        df = pd.read_csv(file)
        drop = list()
        for i, row in df.iterrows():
            if len(row.sgts.split(",")) == 1:
                drop.append(i)

        print("Dropping", len(drop), "out of", df.shape[0])
        df = df.drop(drop)

        if first:
            df.to_csv("dump/Poplulalted_All.csv", index=False)
        else:
            df.to_csv("dump/Poplulalted_All.csv", index=False, mode="a")

def counterfactuals():
    found_sgts = ['muslim', 'younger', 'men', 'heterosexual', 'democrat', 'old',
                  'millenial', 'protestant', 'white', 'asian', 'nonbinary',
                  'middle aged', 'jewish', 'latino', 'buddhist', 'catholic',
                  'middle eastern', 'republican', 'lgbtq', 'elder', 'canadian',
                  'female', 'trans', 'indian', 'conservative', 'women',
                  'american', 'black', 'hispanic', 'taiwanese', 'straight',
                  'bisexual', 'paralyzed', 'arab', 'blind', 'young', 'mexican',
                  'immigrant', 'lgbt', 'chinese', 'communist', 'sikh', 'elderly',
                  'lesbian', 'male', 'older', 'aged', 'christian', 'liberal',
                  'woman', 'man', 'homosexual', 'african', 'gay', 'migrant',
                  'deaf', 'jew', 'english', 'european', 'queer', 'teenage',
                  'spanish', 'japanese', 'transgender']

    p = Preprocessing("data/data_chunks", "extended_SGT.txt", "dump/", found_sgts)
    for i in range(2, 100):
        current_chunk = "dump/Chunk_" + str(i) + "_Populated_All.csv"
        counter = "dump/Counterfactuals_Chunk_" + str(i) + "_Populated_All.csv"
        if Path(counter).is_file():
            continue
        print("working on ", current_chunk)
	try:
            p.generate_counterfactuals(current_chunk)
	except Exception:
            continue
        

counterfactuals()
