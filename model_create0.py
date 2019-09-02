from keras import layers
from keras import models
import keras
from keras import regularizers
from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback


def model_create(filter_size, sliding_window_size, channels):
    model = models.Sequential()
    model.add(layers.GRU(units=32, return_sequences=False,
                          input_shape=(sliding_window_size, channels),
                          kernel_initializer='glorot_normal',recurrent_dropout=0.15, unroll=True))
    model.add(layers.Dense(1, activation='sigmoid'))
    rmsprop = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0)
    model.compile(optimizer=rmsprop, loss='binary_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model



def model_create_frozen(filter_size, sliding_window_size, channels):
    model = models.Sequential()
    model.add(layers.GRU(units=32, return_sequences=False,
                          input_shape=(sliding_window_size, channels),
                          kernel_initializer='glorot_normal', recurrent_dropout=0.15, unroll=True, trainable=False))
    model.add(layers.Dense(1, activation='sigmoid'))

    rmsprop = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0)
    model.compile(optimizer=rmsprop, loss='binary_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model


def model_create_dense():

    model = models.Sequential()
    model.add(layers.Dense(1,input_shape=(32,), activation='sigmoid'))

    rmsprop = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0)
    model.compile(optimizer=rmsprop, loss='binary_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model


model = model_create(6,60,3)