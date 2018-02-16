from gini_index import *

def split_data(index, value, dataset):

    left, right = list(), list()

    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)

    return left, right


def split_get(dataset):

    class_values = list(set(row[-1] for row in dataset))

    b_index, b_value, b_score, b_groups = 999, 999, 999, None

    for index in range(len(dataset[0]) - 1):
        for row in dataset:
            groups = split_data(index, row[index], dataset)
            # print(groups)
            # print('-'*10)
            gini = gini_index(groups, class_values)
            # print(gini)
            # print('**'*10)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index': b_index, 'value': b_value, 'groups': b_groups}

def split(node, max_depth, min_size, depth):
    left, right = node['groups']
    del(node['groups'])
    print("left is ----")
    print(left)
    print("right is ----")
    print(right)
    if not left or not right:
        print("no left or no right")
        node['left'] = node['right'] = to_terminal(left + right)
        return
    
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return

    if len(left) <= min_size:
        node['left'] = to_terminal(left)

    else:
        node['left'] = get_split(left)
        split(node['left'], max_depth, min_size, depth + 1)
    
    if len(right) <= min_size:
        node['right'] = to_terminal(right)

    else:
        node['right'] = get_split(right)
        split(node['right'], max_depth, min_size, depth + 1)