# Gini Index
def gini(data_groups, class_values):

    r = 0.0

    for c in class_values:
        for g in data_groups:
            l = len(g)
            if l == 0:
                continue

            prop = [row[-1] for row in g].count(c) / float(l)
            r += (prop * (1.0 - prop))

    return r


# find the best split point
def split_left_right(index, value, dataset):

    left, right = list(), list()

    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)

    return left, right


# best split point for a dataset
def split_dataset(dataset):

    class_values = list(set(row[-1] for row in dataset))

    index, value, score, groups = 1, 1, 1, None

    for index in range(len(dataset[0]) - 1):
        for row in dataset:
            groups = split_left_right(index, row[index], dataset)

            g = gini(groups, class_values)

            if g < score:
                index, value, score, groups = index, row[index], g, groups

    return {'index': index, 'value': value, 'groups': groups}


# build tree
def tree_to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)


def tree_split(node, max_depth, min_size, depth):
    left, right = node['groups']
    del(node['groups'])
    #print("left is ----")
    #print(left)
    #print("right is ----")
    #print(right)
    if not left or not right:
        #print("no left or no right")
        node['left'] = node['right'] = tree_to_terminal(left + right)
        return

    if depth >= max_depth:
        node['left'], node['right'] = tree_to_terminal(
            left), tree_to_terminal(right)
        return

    if len(left) <= min_size:
        node['left'] = tree_to_terminal(left)

    else:
        node['left'] = split_dataset(left)
        tree_split(node['left'], max_depth, min_size, depth + 1)

    if len(right) <= min_size:
        node['right'] = tree_to_terminal(right)

    else:
        node['right'] = split_dataset(right)
        tree_split(node['right'], max_depth, min_size, depth + 1)


def tree_build(train, max_depth, min_size):
    root = split_dataset(train)
    tree_split(root, max_depth, min_size, 1)
    return root


def tree_print(node, depth=0):
    if isinstance(node, dict):
        print('%s[X%d < %.3f]' %
              ((depth * '  ', (node['index'] + 1), node['value'])))
        #tree_print(node['left'], depth + 1)
        #tree_print(node['right'], depth + 1)
    else:
        print('%s[%s]' % ((depth * '  ', node)))
