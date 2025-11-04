import os
import random
import  numpy as np
import pandas as pd
from numpy.random import sample
from pandas import DataFrame
import glob
from tqdm import tqdm
import librosa
from librosa.util import example_info

#Обробка аудіо і текстів

# def load_tsv(tsv_path:str, data_path:str, example_count:int):
def load_tsv(tsv_path:str, example_count:int):
    # Читання TSV в DataFrame
    df= pd.read_csv(tsv_path, sep='\t')
    # перевірка, чи всі аудіо файли існують
    # df["exists"]=df["path"].apply(lambda p: os.path.exists(os.path.join(data_path, p)))
    # missing=df[~df["exists"]]
    # Head 5000 (1000)
    df=df.head(example_count)
    print(df)
    # Інформація
    print(f"All files: {len(df)}")
    # print(f"Exists file: {df["exists"].sum()}")
    # print(f"Missing files: {len(missing)}")
    # Повернення функції
    return df

def load_tsv_full(tsv_path:str):
    df = pd.read_csv(tsv_path, sep='\t')
    return df
def load_symbols(df:DataFrame):
    unique_symbols = set()
    for line in df.sentence:
        for symbol in line:
            unique_symbols.add(symbol)
    alphabet=sorted(unique_symbols)
    char2idx = {c: i+1 for i, c in enumerate(alphabet)} # 0 = blank \0
    idx2char = {i+1 : c for i, c in enumerate(alphabet)}
    #TODO FIX
    num_classes = len(alphabet) + 1 #Symbol count
    return alphabet, char2idx, idx2char, num_classes

# def optimal_length(data_path: str,example_count: int, num_features: int):
#     files = glob.glob(os.path.join(data_path, "*.mp3"))
#     samples = random.sample(files, min(example_count, len(files)))
#     lengths = []
#     for path in samples:
#         y, sr = librosa.load(path, sr =16000)
#         mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=num_features).T
#         lengths.append(len(mfcc))
#     print(f"n={len(lengths)}")
#     percen = np.percentile(lengths, [50, 75, 90, 95, 99, 100])
#     print(f"Percentile ={percen}")
#     return  int(max(percen))

#Домашнє Завдання
def optimal_length_df(data_path: str, df: DataFrame, num_features: int):
    files = [os.path.join(data_path, f) for f in df.path.values]
    lengths = []
    for path in files:
        y, sr =librosa.load(path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=num_features).T
        lengths.append(len(mfcc))
    return int(np.max(lengths))
#################
# Шляхи до файлів
TSV_PATH = "content/uk/validated.tsv"
DATA_PATH = "content/uk/clips"
#Змінні
EXAMPLE_COUNT = 2
NUM_FEATURES = 40 # Від 13 до 40
#################
DF = load_tsv(TSV_PATH, EXAMPLE_COUNT)
# alphabet, char2idx, idx2char, num_classes = load_symbols(DF)
alphabet, char2idx, idx2char, num_classes = load_symbols(load_tsv_full(TSV_PATH))
# print(alphabet)
# print(char2idx)
# input_length = optimal_length(DATA_PATH, EXAMPLE_COUNT, NUM_FEATURES)
input_length = optimal_length_df(DATA_PATH, DF, NUM_FEATURES)
print(f"Input length ={input_length}")



















