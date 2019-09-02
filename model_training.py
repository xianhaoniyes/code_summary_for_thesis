import numpy as np
from model_create0 import model_create
from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback
import keras
import json
from keras.models import load_model
import matplotlib.pyplot as plt
#############################################################################################
train_auc = []
val_auc = []

class roc_callback(Callback):

    def __init__(self, training_data, validation_data):

        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]


    def on_epoch_end(self, epoch, logs={}):

        y_pred = self.model.predict(self.x)
        roc = roc_auc_score(self.y, y_pred)
        y_pred_val = self.model.predict(self.x_val)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        print('\rroc-auc: %s - roc-auc_val: %s' % (str(round(roc,4)),str(round(roc_val,4))),end=100*' '+'\n')
        train_auc.append(roc)
        val_auc.append(roc_val)
        return

i = 0

test_vectors = np.load('new_all_test_vectors.npy')[i]
test_labels = np.load('new_all_test_labels.npy')[i]

validation_vectors = np.load('new_mix_validation_vectors.npy')[i]
validation_labels = np.load('new_mix_validation_labels.npy')[i]

train_vectors = np.load('new_mix_train_vectors.npy')[i]
train_labels = np.load('new_mix_train_labels.npy')[i]


FILTER_SIZE = 5 # for pure RNN we do not use this.
SLIDING_WINDOW_SIZE = 60
BATCH_SIZE = 512
CHANNELS = 3
filepath = 'model_'+str(i)+'.h5'
model = model_create(FILTER_SIZE, SLIDING_WINDOW_SIZE, CHANNELS)
history = model.fit(train_vectors, train_labels, epochs=200,
                    validation_data=(validation_vectors, validation_labels),
                    batch_size=BATCH_SIZE, shuffle=True,
                    callbacks=[roc_callback(training_data=(train_vectors, train_labels),
                                            validation_data=(validation_vectors, validation_labels)),
                               keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0,
                                                               save_best_only=True, save_weights_only=False,
                                                               mode='auto', period=1)])


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.ylim((0.2, 0.7))
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('history'+str(i))

model = load_model(filepath)  # This part is really important pay attention!
test_auc = []

for j in range(0, len(test_labels)):
    result = model.predict(test_vectors[j])
    test_auc.append(roc_auc_score(test_labels[j], result))

with open('trainHistoryDict_'+str(i)+'.json', 'w') as f:
    json.dump(history.history, f)
print(np.mean(test_auc))
np.save('train_auc_'+str(i)+'.npy', train_auc)
np.save('val_auc_'+str(i)+'.npy', val_auc)
np.save('test_auc_'+str(i)+'.npy', test_auc)
