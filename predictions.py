import numpy as np
from tensorflow.keras.backend import ctc_decode
from tensorflow.keras.models import load_model

###LOAD DATA
import data

#
prediction_model = load_model('model.h5')
#
def decode_batch(pred):
    decoded, _ = ctc_decode(pred, input_length=np.ones(pred.shape[0])*pred.shape[1])
    return decoded[0].numpy()

from data import extract_features, idx2char, input_length, NUM_FEATURES

def recognize_from_file(path):
    mfcc = extract_features(path, NUM_FEATURES, input_length)
    mfcc = np.expand_dims(mfcc, axis=0)
    pred = prediction_model.predict(mfcc)
    decoded = decode_batch(pred)[0]
    text = "".join(idx2char[c] for c in decoded if c > 0)
    return text

#Розпізнання
pred = prediction_model.predict(data.X[:1])
decoded = decode_batch(pred)
for i in range(len(decoded)):
    text = "".join([data.idx2char[c] for c in decoded[i] if c > 0])
    print("Оригінал:", data.DF.iloc[i]["sentence"])
    print("Розпізнано:", text)