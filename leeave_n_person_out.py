import numpy as np
from itertools import combinations


labels = np.load('new_labels_matrix.npy')
person_to_remain = np.load('new_person_to_remain.npy')
labels = labels[person_to_remain]
labels = np.array(labels)

def percentage_cacluation(X):
    length = 0.0
    summy = 0.0
    for i in range(0, len(X)):
        length = length+len(X[i])
        summy = summy+sum(X[i])

    return summy/length

lists_now = []
lists_remain = []

for i in range(0, 6):

    list1 = list(range(0, 8*i))
    list2 = list(range(8*(i+1), 50, 1))
    list_remain = list1+list2
    list_now = list(range(8 * i, 8 * (i + 1)))

    lists_remain.append(list_remain)
    lists_now.append(list_now)


candidate_possible_tests = []
candidate_possible_trains = []

for i in range(0, len(lists_now)):
    possible_change_combination = list(combinations(lists_remain[i], 2))

    for j in range(0, len(possible_change_combination)):
        list_test = lists_now[i] + list(possible_change_combination[j])
        list_train = list(set(lists_remain[i]) - set(possible_change_combination[j]))

        candidate_possible_tests.append(list_test)
        candidate_possible_trains .append(list_train)

test_set = []
train_set = []


for i in range(0, len(candidate_possible_tests)):

    test_list = candidate_possible_tests[i]

    train_list = candidate_possible_trains[i]

    if abs(percentage_cacluation(labels[test_list])-percentage_cacluation(labels[train_list])) < 0.015:
        test_set.append(candidate_possible_tests[i])
        train_set.append(candidate_possible_trains[i])



# for item in test_set:
#     print(item)
# print(len(test_set))
count = [0, 0, 0, 0, 0]
for i in range(0, len(test_set)):
    if test_set[i][0] == 0:
        if test_set[i][8] == 31 and test_set[i][9] == 45:
            count[0] = i
    if test_set[i][0] == 16:
        if test_set[i][8] == 8 and test_set[i][9] == 13:
            count[1] = i
    if test_set[i][0] == 24:
        if test_set[i][8] == 14 and test_set[i][9] == 48:
            count[2] = i
    if test_set[i][0] == 32:
        if test_set[i][8] == 9 and test_set[i][9] == 13:
            count[3] = i
    if test_set[i][0] == 40:
        if test_set[i][8] == 48 and test_set[i][9] == 49:
            count[4] = i


test_set = [test_set[count[0]], test_set[count[1]], test_set[count[2]], test_set[count[3]], test_set[count[4]]]
train_set = [train_set[count[0]], train_set[count[1]], train_set[count[2]], train_set[count[3]], train_set[count[4]]]
# print(percentage_cacluation(labels[test_set[0]]), percentage_cacluation(labels[train_set[0]]))
#
# current_test_labels = labels[test_set[0]]
# current_train_labels =labels[train_set[0]]
# current_test_labels = np.concatenate(current_test_labels)
# current_train_labels = np.concatenate(current_train_labels)
# print(np.sum(current_test_labels)/len(current_test_labels), np.sum(current_train_labels)/len(current_train_labels))
print(test_set)
#
# np.save('new_test_set.npy', test_set)
# np.save('new_remain_set.npy', train_set)








