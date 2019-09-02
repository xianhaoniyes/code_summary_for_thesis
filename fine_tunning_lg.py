from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
import numpy as np


f = 0

labels = np.load('new_labels_matrix.npy')
train_set = np.load('new_remain_set.npy')[f]
person_to_remain = np.load('new_person_to_remain.npy')

vectors = np.load('fold_vectors_'+str(f)+'.npy')
vectors = vectors[train_set]


labels = labels[person_to_remain]
labels = labels[train_set]
fold_weights_cv = np.zeros((40, 25))
for i in range(0, len(vectors)):
    lg = LogisticRegressionCV(solver='sag', cv=3, scoring='roc_auc',class_weight='balanced', max_iter=80000)
    lg.fit(vectors[i], labels[i])
    fold_weights_cv[i, 0:24] = lg.coef_
    fold_weights_cv[i, 24] = lg.intercept_

np.save('fold_weights_lg_'+str(f)+'.npy', fold_weights_cv)
