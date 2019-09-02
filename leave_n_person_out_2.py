import numpy as np
from itertools import combinations
labels = np.load('new_labels_matrix.npy')
person_to_remain = np.load('new_person_to_remain.npy')
labels = labels[person_to_remain]

def percentage_cacluation(X):
    length = 0.0
    summy = 0.0
    for i in range(0, len(X)):
        length = length+len(X[i])
        summy = summy+sum(X[i])

    return summy/length


test_set = np.load('new_test_set.npy')
remain_set = np.load('new_remain_set.npy')
total_validation_set = []
total_train_set = []

for i in range(0, len(remain_set)):

    k_fold_remain_set = list(remain_set[i])
    lists_remain = []
    lists_now = []
    for j in range (0, 5):
        list1 = k_fold_remain_set[0: 8 * j]
        list2 = k_fold_remain_set[8 * (j + 1): 40]
        list_remain = list1 + list2
        list_now = k_fold_remain_set[4 * j: 4 * (j + 1)]

        lists_remain.append(list_remain)
        lists_now.append(list_now)

    candidate_possible_validations = []
    candidate_possible_trains = []

    for k in range(0, len(lists_now)):

        possible_change_combination = list(combinations(lists_remain[k], 2))

        for f in range(0, len(possible_change_combination)):
            list_validation = lists_now[k] + list(possible_change_combination[f])
            list_train = list(set(lists_remain[k]) - set(possible_change_combination[f]))

            candidate_possible_validations.append(list_validation)
            candidate_possible_trains.append(list_train)

    validation_set = []
    train_set = []

    for z in range(0, len(candidate_possible_validations)):

        validation_list = candidate_possible_validations[z]
        train_list = candidate_possible_trains[z]

        if abs(percentage_cacluation(labels[validation_list])-percentage_cacluation(labels[train_list]))< 0.001:
            validation_set.append(candidate_possible_validations[z])
            train_set.append(candidate_possible_trains[z])

    total_validation_set.append(validation_set)
    total_train_set.append(train_set)


for i in range(0, len(total_validation_set)):
    for j in range(0, len(total_validation_set[i])):
        print(total_validation_set[i][j])
    print("fold"+str(i))

# real_validation_set = [total_validation_set[0][0], total_validation_set[1][0], total_validation_set[2][0],
#                        total_validation_set[3][0], total_validation_set[4][0]]
#
# real_train_set = [total_train_set[0][0], total_train_set[1][0], total_train_set[2][0],
#                     total_train_set[3][0], total_train_set[4][0]]

#
# np.save('new_validation_set.npy', real_validation_set)
# np.save('new_train_set.npy', real_train_set)

