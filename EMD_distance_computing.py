import numpy as np
from EMD import ComputeKernelEMD
from EMD import ComputeKernelEMD1D
from EMD import ComputeKernelEMD_weights



for f in range(0, 5):
    vectors = np.load('fold_clustering_vectors_'+str(f)+'.npy')
    weights = np.load('fold_clustering_weights_'+str(f)+'.npy')

    distances = ComputeKernelEMD_weights(vectors, vectors, weights, weights, 'sqeuclidean')

    np.save('clustering_distance_matrix'+str(f)+'.npy', distances)