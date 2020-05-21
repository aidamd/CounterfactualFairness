import os
import pandas as pd
from tqdm import tqdm
from helpers import read_sgts
import re


class Preprocessing():

    def __init__(self, predict_dir, sgt_path, result_dir):
        self.predict_dir = predict_dir
        self.sgt_file_path = sgt_path
        self.sgts = read_sgts(sgt_path)
        self.result_dir = result_dir

    def extract_posts_with_sgt(self):

        for filename in os.listdir(self.predict_dir):
            print("--------------------------------")
            print(os.path.join(self.predict_dir, filename))
            print("--------------------------------")

            dataset = pd.read_csv(os.path.join(self.predict_dir, filename))
            rows_list = []
            for index, row in tqdm(dataset.iterrows()):
                text = row.text.lower()
                for sgt in self.sgts:
                     if re.findall("((^|[^\w])(" + sgt + ")([^\w]|$|s|es))", text):
                    #if sgt in text:
                        row_dict = dict(row)
                        row_dict["detected_sgt"] = sgt
                        rows_list.append(row_dict)
                        break
            sgt_data_df = pd.DataFrame(rows_list)
            result_path = os.path.join(self.result_dir, "SGT_" + filename)
            print("saving to", result_path)
            sgt_data_df.to_csv(result_path)

    def extract_all_sgts(self):
        for filename in os.listdir(self.result_dir):
            if not filename.startswith("SGT_"):
                continue
            print("--------------------------------")
            print(os.path.join(self.result_dir, filename))
            print("--------------------------------")
            dataset = pd.read_csv(os.path.join(self.result_dir, filename))
            rows_list = []
            for index, row in tqdm(dataset.iterrows()):
                text = row.text.lower()
                present_sgts = [sgt for sgt in self.sgts if
                                re.findall("((^|[^\w])(" + sgt + ")([^\w]|$|s|es))", text)]
                present_sgts_str = ','.join(present_sgts)
                row_dict = dict(row)
                row_dict["sgts"] = present_sgts_str
                rows_list.append(row_dict)

            sgt_data_df = pd.DataFrame(rows_list)
            result_path = os.path.join(self.result_dir, "Populated_" + filename)
            print("saving to", result_path)
            sgt_data_df.to_csv(result_path)

    def merge_preprocessed_chunks(directory):
        os.chdir(directory)
        all_filenames = [i for i in os.listdir()]
        combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames])
        combined_csv.to_csv("SGT-Gab.csv", index=False, encoding='utf-8-sig')