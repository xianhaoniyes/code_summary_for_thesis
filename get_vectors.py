import numpy as np
from scipy import stats


def get_vectors(acc, speaking):
    ##raw_vectors

    accel_x = stats.zscore(acc[0][0][:, 0])
    accel_y = stats.zscore(acc[0][0][:, 1])
    accel_z = stats.zscore(acc[0][0][:, 2])

    absAccelX = np.absolute(accel_x)
    absAccelY = np.absolute(accel_y)
    absAccelZ = np.absolute(accel_z)

    accel = np.array([accel_x, accel_y, accel_z])
    abs_accel = np.array([absAccelX, absAccelY, absAccelZ])
    accelmag = np.sqrt(pow(accel_x, 2) + pow(accel_y, 2) + pow(accel_z, 2))

    edge = 36000
    accel = accel[:, 0:edge]
    abs_accel = abs_accel[:, 0:edge]
    accelmag = accelmag[0:edge]
    speaking = speaking[0:edge]
    window_time = 3
    window_size = 20 * window_time
    overlap = 30
    lower_bound = 0
    upper_bound = window_size

    mag_vectors = []
    abs_vectors = []
    vectors = []
    labels = []

    while upper_bound <= edge:

        partial_accel = np.array(accel[:, lower_bound: upper_bound])
        partial_mag = np.array(accelmag[lower_bound:upper_bound])
        partial_abs = np.array(abs_accel[:, lower_bound:upper_bound])

        vectors.append(partial_accel)
        mag_vectors.append(partial_mag)
        abs_vectors.append(partial_abs)

        label = round(sum(speaking[lower_bound:upper_bound]) / window_size)
        labels.append(label)
        lower_bound = upper_bound - overlap
        upper_bound = lower_bound + window_size

    vectors = np.array(vectors)
    labels = np.array(labels)
    return vectors, mag_vectors, abs_vectors, labels

