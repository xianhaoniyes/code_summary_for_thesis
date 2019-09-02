import numpy as np
from sklearn.model_selection import train_test_split

test_set = np.load('new_test_set.npy')
remain_set = np.load('new_remain_set.npy')


vectors = np.load('new_vectors_matrix.npy')
labels = np.load('new_labels_matrix.npy')
person_to_remain = np.load('new_person_to_remain.npy')

vectors = vectors[person_to_remain]
labels = labels[person_to_remain]


all_test_vectors = []
all_test_labels = []

all_train_vectors = []
all_train_labels = []

all_validation_vectors = []
all_validation_labels = []

for i in range(0, len(test_set)):

    test_vectors = vectors[test_set[i]]
    test_labels = labels[test_set[i]]

    all_test_vectors.append(test_vectors)
    all_test_labels.append(test_labels)

    remain_vectors = vectors[remain_set[i]]
    remain_labels = labels[remain_set[i]]
    remain_vectors = np.vstack(remain_vectors)
    remain_labels = np.concatenate(remain_labels, axis=0)

    train_vectors, validation_vectors, train_labels, validation_labels = \
        train_test_split(remain_vectors, remain_labels, test_size= 0.25, shuffle=True, stratify=remain_labels)

    all_train_vectors.append(train_vectors)
    all_train_labels.append(train_labels)

    all_validation_vectors.append(validation_vectors)
    all_validation_labels.append(validation_labels)


print(np.shape(all_test_vectors))
print(np.shape(all_validation_vectors))
print(np.shape(all_train_vectors))


np.save('new_all_test_vectors.npy', all_test_vectors)
np.save('new_all_test_labels.npy', all_test_labels)

np.save('new_mix_validation_vectors.npy', all_validation_vectors)
np.save('new_mix_validation_labels.npy', all_validation_labels)

np.save('new_mix_train_vectors.npy', all_train_vectors)
np.save('new_mix_train_labels.npy', all_train_labels)

