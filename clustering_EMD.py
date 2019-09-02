import numpy as np
# from EMD import ComputeKernelEMD
# from EMD import ComputeKernelEMD1D
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plot


for f in range(0, 5):
    vectors = np.load('fold_vectors_'+str(f)+'.npy')
    fold_weights = []
    fold_clustering_vectors = []
    for i in range(0, 50):
        weights = []
        current_vectors = vectors[i]

        ms = KMeans(n_clusters=6)
        ms.fit(current_vectors)
        centers = ms.cluster_centers_

        labels = ms.labels_

        fold_clustering_vectors.append(centers)

        for k in range(0,len(centers)):
            weights.append(len(current_vectors[labels == k]))

        weights = weights[:]/np.sum(weights)

        fold_weights.append(weights)

    np.save('fold_clustering_vectors_'+str(f)+'.npy', fold_clustering_vectors)
    np.save('fold_clustering_weights_'+str(f)+'.npy', fold_weights)



