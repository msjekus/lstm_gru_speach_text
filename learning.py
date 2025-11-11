import tensorflow as tf
import numpy as np
from tensorflow.keras.backend import ctc_batch_cost
from tensorflow.keras.layers import Lambda, Dense, Dropout,BatchNormalization, Input, Bidirectional, LSTM, GRU
from  tensorflow.keras.models import Model
### load data
import data

import numpy as np

NPZ_PATH = "X_Y_padded_input_lengths_label_lengths.npz"
data = np.load(NPZ_PATH)

DATA_COUNT = 5000  # або 1000 якщо слабкий ПК
X = data["X"][:DATA_COUNT]
Y_padded = data["Y_padded"][:DATA_COUNT]
input_lengths = data["input_lengths"][:DATA_COUNT]
label_lengths = data["label_lengths"][:DATA_COUNT]

import os
### Побудова моделі
# Input
input_data = Input(shape=(None, data.NUM_FEATURES), name='input')
labels = Input(shape=(None,), name='Labels')
input_len = Input(shape=(1,), name='input_len')
label_len = Input(shape=(1,), name='label_len')
#Мережа
x = Bidirectional(GRU(512, return_sequences=True))(input_data)
x = BatchNormalization ()(x)
x = Bidirectional(GRU(512, return_sequences=True))(x)
x = BatchNormalization ()(x)
x = Bidirectional(GRU(256, return_sequences=True))(x)
x = Dropout(0.3)(x)
x = Dense(data.num_classes, activation='softmax')(x)
#CTC
def ctc_loss_lambda_func(y_true, y_pred, input_length, label_length):
    return ctc_batch_cost(y_true, y_pred, input_length, label_length)
lost_out = Lambda(
    lambda args: ctc_loss_lambda_func(*args), output_shape=(1,), name="ctc"
)([labels, x, input_len, label_len])
#Модель

model = Model(inputs=[input_data, labels, input_len, label_len], outputs= lost_out)
model.compile(optimizer='adam', loss=lambda y_true, y_pred:y_pred)
print(model.summary())

CHECKPOINT_WEIGHTS = "checkpoint.weights.h5"

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    CHECKPOINT_WEIGHTS, save_weights_only=True, save_best_only=False, verbose=1
)

# if os.path.exists(CHECKPOINT_WEIGHTS):
#     print("Found checkpoint weights, loading...")
#     try:
#         model.load_weights(CHECKPOINT_WEIGHTS)
#         print("Weights loaded from checkpoint.")
#     except Exception as e:
#         print("Could not load weights:", e)
#
# # Підготовка даних для fit
# # labels у нас вже вирівняні у Y_padded
# # input_lengths треба подати у вигляді (N,1)
# input_lengths_fit = data.input_length.reshape((-1, 1))
# label_lengths_fit = data.label_lengths.reshape((-1, 1)

#Навчання
PREDICTION_MODEL_PATH = "model.h5"
BATCH_SIZE= 8
history = model.fit(
    x=[data.X, data.Y_padded, data.input_lengths, data.label_lengths],
    y=np.zeros(len(data.X)),
    batch_size=BATCH_SIZE,
    epochs=3, #15-20, 30-50, 100-300
    validation_split=0.1,
)
#Створення моделі для передбачення
prediction_model = Model(inputs=input_data, outputs=x)
prediction_model.save('model.h5')
print("Saved prediction model to", PREDICTION_MODEL_PATH)