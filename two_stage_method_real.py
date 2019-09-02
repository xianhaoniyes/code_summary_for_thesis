import numpy as np
from W_matrix_computing import W_matrix_computing

from sklearn.metrics import roc_auc_score
import cvxpy as cp
from sklearn.linear_model import LogisticRegression
from W_matrix_computing import marginal_weights_computing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV

labels = np.load('new_labels_matrix.npy')
person_to_remain = np.load('new_person_to_remain.npy')
labels = labels[person_to_remain]

f = 1


test_set = np.load('new_test_set.npy')[f]
remain_set = np.load('new_remain_set.npy')[f]
fold_vectors = np.load('fold_vectors_' + str(f) + '.npy')
fold_weights = np.load('fold_weights_lg_'+str(f)+'.npy')
fold_vectors_train = fold_vectors[remain_set]
fold_labels_train = labels[remain_set]
fold_auc = []
for j in range(0, 10):

    person = test_set[j]
    fold_vectors_i = fold_vectors[person]
    labels_i = labels[person]

    #### this part is for the first stage, conditional_stage
    # W_matrix = W_matrix_computing(fold_vectors_i)
    #
    # length = len(fold_vectors_i)
    #
    # D_matrix = np.eye(length, length)
    #
    # for i in range(0, length):
    #     D_matrix[i, i] = (np.sum(W_matrix[i, :]))**(-0.5)
    #
    # I = np.eye(length, length)
    #
    # Lu = I - np.dot((np.dot(D_matrix, W_matrix)), D_matrix)
    #
    # prediction = []
    # for i in range(0, len(fold_vectors_train)):
    #     pre = fold_weights[i, 0:24].dot(np.transpose(fold_vectors_i)) \
    #               + fold_weights[i, 24]
    #
    #     pre = np.squeeze(pre)
    #
    #     pre = 1 / (1 + np.exp(-pre))
    #
    #     prediction.append(pre)
    #
    # prediction = np.array(prediction)
    # prediction = np.squeeze(prediction)
    # prediction = np.transpose(prediction)
    #
    # p_matrix = np.dot(np.dot(np.transpose(prediction), Lu), prediction)
    #
    #
    #
    # # 下面这段代码非常关键
    # Q = p_matrix
    # beta = cp.Variable(40)
    # A = np.ones(40)
    # G = np.diagflat(np.ones(40))
    # h = np.ones(40)*0.01
    #
    # obj = cp.quad_form(beta, Q)
    # problem = cp.Problem(cp.Minimize(obj), [A.T@beta == 1, G@beta >= h])
    # problem.solve()
    # importance = beta.value
    # importance = np.squeeze(importance)
    #
    # print(np.argmax(importance))
    # train_vectors = np. vstack(fold_vectors_train)
    #
    # train_labels = np.concatenate(fold_labels_train)
    #
    # weights = []
    #
    # for i in range(0, 40):
    #     current_weights = (importance[i]/len(fold_vectors_train[i])) * np.ones(len(fold_vectors_train[i]))
    #     weights.append(current_weights)
    #
    # weights = np.array(weights)
    # weights = np.concatenate(weights)




    ############################################# first part over ###############################################

    #################################### this part is for the marginal stage ####################################

    marginal_weights = marginal_weights_computing(fold_vectors_i, fold_vectors_train)
    #
    weights = marginal_weights

    cv = LogisticRegressionCV(penalty='l2', cv = 5, solver = 'sag', max_iter=10000)

    cv.fit(train_vectors, train_labels, weights)

    res = cv.predict_proba(fold_vectors_i)
    item_auc = roc_auc_score(labels_i, res[:, 1])
    print(item_auc)
    fold_auc.append(item_auc)


#
# np.save('fold_cvxpy_'+str(f)+'.npy', fold_auc)