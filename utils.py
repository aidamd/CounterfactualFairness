from nltk import tokenize as nltk_token
import numpy as np
import operator
import re
from sklearn.metrics import f1_score, precision_score, recall_score
from collections import Counter
import random

def preprocess(df):
    print(df.shape[0], "datapoints in dataset")
    df = tokenize_data(df, "text")
    df = remove_empty(df, "text")
    print(df.shape[0], "datapoints after removing empty strings")
    return df

def tokenize_data(corpus, col):
    for idx, row in corpus.iterrows():
        corpus.at[idx, col] = nltk_token.WordPunctTokenizer().tokenize(clean(row[col]))
    return corpus

def remove_empty(corpus, col):
    drop = list()
    for i, row in corpus.iterrows():
        if row[col] == "" or len(row[col]) < 4 or len(row[col]) > 100:
            drop.append(i)
    return corpus.dropna().drop(drop)

def clean(sent):
    http = re.sub("https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=/]{2,256}"
                  "\.[a-z]{2,4}([-a-zA-Z0-9@:%_\+.~#?&//=]*)", "", sent)
    return re.sub(r'[^a-zA-Z ]', r'', http.lower())

def learn_vocab(corpus, vocab_size):
    print("Learning vocabulary of size %d" % (vocab_size))
    tokens = dict()
    for sent in corpus:
        for token in sent:
            if token in tokens:
                tokens[token] += 1
            else:
                tokens[token] = 1
    words, counts = zip(*sorted(tokens.items(), key=operator.itemgetter(1), reverse=True))
    return list(words[:vocab_size]) + ["<unk>", "<pad>"]


def tokens_to_ids(corpus, vocab):
    #print("Converting corpus of size %d to word indices based "
    #      "on learned vocabulary" % len(corpus))
    if vocab is None:
        raise ValueError("learn_vocab before converting tokens")

    mapping = {word: idx for idx, word in enumerate(vocab)}
    unk_idx = vocab.index("<unk>")

    for i, row in enumerate(corpus):
        for j, tok in enumerate(row):
            try:
                corpus[i][j] = mapping[corpus[i][j]]
            except:
                corpus[i][j] = unk_idx
    return corpus


def load_embedding(vocabulary, file_path, embedding_size):
    embeddings = np.random.randn(len(vocabulary), embedding_size)
    found = 0
    with open(file_path, "r") as f:
        for line in f:
            split = line.split()
            idx = len(split) - embedding_size
            vocab = "".join(split[:idx])
            if vocab in vocabulary:
                embeddings[vocabulary.index(vocab)] = np.array(split[idx:], dtype=np.float32)
                found += 1
    print("Found {}/{} of vocab in word embeddings".
          format(found, len(vocabulary)))
    return embeddings

def get_batches(data, data_idx, batch_size, pad_idx, hate=None, counter=None):
    batches = []
    for idx in range(len(data) // batch_size + 1):
        if idx * batch_size !=  len(data):
            data_batch = data[idx * batch_size: min((idx + 1) * batch_size, len(data))]
            idx_batch = data_idx[idx * batch_size: min((idx + 1) * batch_size, len(data))]
            hate_batch = hate[idx * batch_size: min((idx + 1) * batch_size, len(hate))] \
                if hate else None

            data_info = batch_to_info(data_batch, hate_batch, idx_batch, pad_idx, counter)
            batches.append(data_info)
    return batches

def batch_to_info(batch, hate, idx, pad_idx, cf):
    max_len = max(len(sent) for sent in batch)
    batch_info = list()
    for i, sent in enumerate(batch):
        padding = [pad_idx] * (max_len - len(sent))
        sentence = {
            "input": sent + padding,
            "counter": [sent + padding] if idx[i] not in cf else
                        [c + padding for c in cf[idx[i]]],
            "length": len(sent),
            "hate": hate[i] if hate else None,
        }
        batch_info.append(sentence)
    return batch_info

def prediction_results(df, pred, label="hate"):
    y = df[label].values.tolist()
    print(": F1 score:", f1_score(y, pred),
          ", Precision:", precision_score(y, pred),
          ", Recall:", recall_score(y, pred)
          )
    print(Counter(y))
    print(Counter(pred))

