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
     

def rank_original_post(beginning_idex, end_index):
	print("working on", beginning_index, "to", end_index - 1)
	result_path = "dump/Ranking_Chunks_" + str(beginning_index)+"_to_"+ str(end_index-1) + ".csv"

	row_list = []
	for i in range(beginning_index, end_index):
	    counter = "dump/Counterfactuals_Chunk_" + str(i) + "_Populated_All.csv"
	    print("Processing", counter)
	    try:
		counter_data = pd.read_csv(counter)
	    except Exception:
		print("bad file!", counter)
		continue

	    for name, group in counter_data.groupby(["id"]):
		# id, text, original_SGT, rank
		record = dict()
		record["id"] = name
		original_record = group.loc[group['original_sgt'] == group['new_sgt']]
		try:
		    record["text"] = original_record["text"].values[0]
		    record["original_SGT"] = original_record["original_sgt"].values[0]
		    original_perplexity = original_record["perplexity"].values[0]
		except Exception:
		    print("faulty record!  id=", name)
		    print(original_record)
		    continue
		perplexities = [group["perplexity"].tolist()[i] for i in range(group.shape[0])]
		perplexities.sort()
		record["rank"] = perplexities.index(original_perplexity)+1
		row_list.append(record)

	perplexity_rank_df = pd.DataFrame(row_list)   
	print("saving to", result_path)
	perplexity_rank_df.to_csv(result_path, index=False, encoding='utf-8-sig')

counterfactuals()
