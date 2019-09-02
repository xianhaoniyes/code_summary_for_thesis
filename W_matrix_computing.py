import numpy as np
import cvxpy as cp


def W_matrix_computing(test_vectors):

    length = len(test_vectors)
    W = np.zeros((length, length))
    lam = 1

    for i in range(0, length):
        for j in range(i, length):

            W[i, j] = np.exp(-np.square(np.linalg.norm(test_vectors[i]-test_vectors[j]))/(2*np.square(lam)))

    W = W + np.transpose(np.triu(W, k=1))
    return W


def marginal_weights_computing(fold_vectors_i, fold_vectors_train):

    lam = 1

    marginal_weights_matrix = []


    for i in range(0,len(fold_vectors_train)):

        little_k_matrix = np.zeros(len(fold_vectors_train[i]))
        K_matrix = W_matrix_computing(fold_vectors_train[i])


        for j in range(0,len(fold_vectors_train[i])):

            a = np.sum(np.exp(-np.square(np.linalg.norm(fold_vectors_train[i][j]-fold_vectors_i, axis=1))/(2*np.square(lam))))
            little_k_matrix[j] = a

        little_k_matrix = len(fold_vectors_train[i]/len(fold_vectors_i))*little_k_matrix

        A = np.ones(len(fold_vectors_train[i]))
        G = np.diagflat(np.ones(len(fold_vectors_train[i])))
        h_low = np.ones(len(fold_vectors_train[i]))*0.0001
        h_high =  np.ones(len(fold_vectors_train[i]))*3

        dada = np.sqrt(len(fold_vectors_train[i]))

        # b = np.ones(len(fold_vectors_train[i]))
        alpha = cp.Variable(len(fold_vectors_train[i]))

        problem = cp.Problem(cp.Minimize(0.5*cp.quad_form(alpha, K_matrix)-little_k_matrix.T@alpha),
                             [G@alpha >= h_low, G@alpha <= h_high,
                              cp.abs(cp.sum(A.T@alpha) - len(fold_vectors_train[i]))
                              <= 3 * 3*dada])
        problem.solve(max_iter = 100000)

        # print(np.sum(alpha.value))
        # print(np.max(alpha.value))
        # print(alpha.value)

        print('solve one')

        importance = alpha.value
        importance = np.squeeze(importance)

        marginal_weights_matrix.append(importance)

    marginal_weights_matrix = np.concatenate(marginal_weights_matrix)

    return marginal_weights_matrix

