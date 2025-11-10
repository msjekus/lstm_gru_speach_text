import os
import random
import  numpy as np
import pandas as pd
from numpy.random import sample
from pandas import DataFrame
import glob

from tensorflow.keras.models import load_model
from tqdm import tqdm
import librosa
from librosa.util import example_info

#Обробка аудіо і текстів

symbols = "абвгдежзийклмнопрстуфхчцшщьюяєіїґ'- "

def clean_text(text:str):
    text=text.lower()
    cleaned_text = ''.join([ch for ch in text if ch in symbols])
    return cleaned_text

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

def extract_features(path:str, num_features: int, input_length:int):
    y,sr = librosa.load(path, sr=16000)
    mfcc=librosa.feature.mfcc(y=y, sr=sr, n_mfcc=num_features).T
    if len(mfcc)<input_length:
        mfcc = np.pad(mfcc, ((0, input_length-len(mfcc)),(0,0)),mode="constant")
    else:
        mfcc=mfcc[:input_length, :]
    return mfcc

def encode_text(text:str,char2idx:dict):
    text=text.lower().strip()
    return [char2idx[c] for c in text if c in char2idx]

def fill_data(data_path: str, df:DataFrame, num_features: int, input_length, char2idx: dict):
    X, Y, input_lengths, label_lengths = [] ,[], [], []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        file_path = os.path.join(data_path, row.path)
        if not os.path.exists(file_path):
            continue
        mfcc=extract_features(file_path, num_features, input_length)
        label = encode_text(row.sentence, char2idx)
        if len(label) == 0:
            continue
        X.append(mfcc)
        Y.append(label)
        input_lengths.append(len(mfcc))
        label_lengths.append(len(label))
    #Вирівнювання
    max_input_len = np.max(input_lengths)
    max_label_len = np.max(label_lengths)
    Y_padded = np.zeros((len(Y), max_label_len))
    for i in range(len(Y)):
        Y_padded[i, :len(Y[i])]= Y[i]
    X = np.array(X)
    input_lengths = np.array(input_lengths)
    label_lengths = np.array(label_lengths)
    return X, Y, Y_padded, input_lengths, label_lengths

def decode_prediction(pred, idx2char):
    text = ""
    prev = -1
    for p in pred:
        idx = np.argmax(p)
        if idx != prev and idx != 0:
            text += idx2char[idx]
        prev = idx
    return text

def recognize_audio(path, model, num_features, input_length, idx2char):
    mfcc = extract_features(path, num_features, input_length)
    mfcc = np.expand_dims(mfcc, axis=0)
    pred = model.predict(mfcc)[0]
    return decode_prediction(pred, idx2char)

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
X, Y, Y_padded, input_lengths, label_lengths = fill_data(
    DATA_PATH, DF, NUM_FEATURES, input_length, char2idx)
print(X, Y, Y_padded, input_lengths, label_lengths)
print("X shape:", X.shape)

model = load_model("model.h5", compile=False)

test_file = os.path.join(DATA_PATH, DF.path.values[0])
recognized = recognize_audio(test_file, model, NUM_FEATURES, input_length, idx2char)

print(f"Original: {clean_text(DF.sentence.values[0])}")
print(f"Recognized: {recognized}")