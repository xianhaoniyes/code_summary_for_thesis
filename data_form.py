from scipy.io import loadmat
import numpy as np
import get_vectors
mat_file = loadmat('Mingle_30minSeg.mat')
data = mat_file['Mingle_30min']
labels = np.genfromtxt('new_LABELS.csv', delimiter=",")


PERSON = [[1, 2, 4, 7, 11, 12, 13, 14, 15, 16, 19, 20, 21, 22, 23, 24, 25, 27, 28, 29, 30, 31],
          [1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 23, 24, 26, 27],
          [2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 15, 17, 18, 19, 20, 22, 24, 25, 26, 29]]

vectors_matrix_all = []
mag_vectors_matrix_all = []
abs_vectors_matrix_all = []
labels_matrix_all = []

for DAY in range(0, 3):
    one_day_data = data[0][DAY]
    vectors_matrix = []
    mag_vectors_matrix = []
    abs_vectors_matrix = []
    labels_matrix = []
    for person in PERSON[DAY]:
        acc = one_day_data[0][person]['accel']
        if DAY == 0:
            speaking = np.array(labels[:, person * 9 + 3])
        if DAY == 1:
            speaking = np.array(labels[:, (person + 32) * 9 + 3])
        if DAY == 2:
            speaking = np.array(labels[:, (person + 62) * 9 + 3])

        vectors, mag_vectors, abs_vectors, vector_labels = get_vectors.get_vectors(acc, speaking)
        vectors_matrix.append(vectors)
        mag_vectors_matrix.append(mag_vectors)
        abs_vectors_matrix.append(abs_vectors)
        labels_matrix.append(vector_labels)

    vectors_matrix_all.append(vectors_matrix)
    mag_vectors_matrix_all.append(mag_vectors_matrix)
    abs_vectors_matrix_all.append(abs_vectors_matrix)
    labels_matrix_all.append(labels_matrix)


vectors_matrix_all = np.vstack(vectors_matrix_all[:])
mag_vectors_matrix_all = np.vstack(mag_vectors_matrix_all[:])
abs_vectors_matrix_all = np.vstack(abs_vectors_matrix_all[:])
labels_matrix_all = np.vstack(labels_matrix_all[:])

vectors_matrix_all = vectors_matrix_all.astype('float32')
labels_matrix_all = labels_matrix_all.astype('int')

vectors_matrix_all = vectors_matrix_all.transpose((0, 1, 3, 2))

print(vectors_matrix_all.shape)
print(mag_vectors_matrix_all.shape)
print(abs_vectors_matrix_all.shape)
print(labels_matrix_all.shape)


np.save('vectors_matrix.npy', vectors_matrix_all)
np.save("mag_vectors_matrix.npy", mag_vectors_matrix_all)
np.save("abs_vectors_matrix.npy", abs_vectors_matrix_all)
np.save('labels_matrix.npy', labels_matrix_all)
