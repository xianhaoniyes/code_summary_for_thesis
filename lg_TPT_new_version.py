import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
f = 4
# test_features = np.load('neural_test_features_fold_'+str(f)+'.npy')
labels = np.load('new_labels_matrix.npy')
person_to_remain = np.load('new_person_to_remain.npy')
labels = labels[person_to_remain]


test_set = np.load('new_test_set.npy')[f]
train_set = np.load('new_remain_set.npy')[f]
labels = labels[test_set]

# test_features = np.load('neural_test_features_fold_'+str(f)+'.npy')

features = np.load('fold_vectors_'+str(f)+'.npy')
test_features = features[test_set]

# train_distance_matrix = np.load('neural_train_distances_matrix_fold_'+str(f)+'.npy')
# test_distance_matrix = np.load('neural_test_distances_matrix_fold_'+str(f)+'.npy')

# distance_matrix = np.load('normal_distances_matrix'+str(f)+'.npy')
#distance_matrix = np.load('clustering_distance_matrix'+str(f)+'.npy')
distance_matrix = np.load('clustering_distance_matrix'+str(f)+'.npy')
train_distance_matrix = distance_matrix[train_set]
train_distance_matrix = train_distance_matrix[:,train_set]
test_distance_matrix = distance_matrix[test_set]
test_distance_matrix = test_distance_matrix[:,train_set]

params = np.load('fold_weights_lg_'+str(f)+'.npy')


# this part is for normal EMD
results_normal_emd = []
auc = []
for i in range(0, 10):
    current_matrix = np.zeros((len(train_distance_matrix)+1, len(train_distance_matrix)+1))

    current_matrix[0, 0] = 0
    current_matrix[1:, 1:] = train_distance_matrix
    current_matrix[0, 1:] = test_distance_matrix[i]
    current_matrix[1:, 0] = test_distance_matrix[i]
    new_classifier = []
    current_kernel_matrix = np.exp((-1 / np.mean(current_matrix[np.nonzero(current_matrix)])) * current_matrix)
    # current_kernel_matrix = np.exp(-0.1 * current_matrix)
    # current_kernel_matrix = current_matrix
    for j in range(0, 25):

        alphas = (2 * np.logspace(-15, 15, 30, base=2)) ** -1
        krr = KernelRidge(kernel='precomputed')
        clf = GridSearchCV(estimator=krr, param_grid=dict(alpha=alphas), cv=5, scoring='neg_mean_absolute_error')
        clf.fit(current_kernel_matrix[1:, 1:], params[:, j])
        coefTestKRR = clf.predict(current_kernel_matrix[0, 1:].reshape(1, -1))
        new_classifier.append(coefTestKRR)

    coefTestKRR = np.array(new_classifier)
    coefTestKRR = np.transpose(coefTestKRR)
    yhatKRR = coefTestKRR[:, 0:24].dot(np.transpose(test_features[i])) \
              + coefTestKRR[:, 24]

    yhatKRR = np.squeeze(yhatKRR)
    result_item = roc_auc_score(labels[i], yhatKRR)
    auc.append(result_item)



np.save('emd_cluster_TPT_auc'+str(f)+'.npy',auc)