
def gini_index(groups, class_values):
    
    gini = 0.0
    
    for c in class_values:
        for g in groups:
            l = len(g)
            if l == 0:
                continue

            prop = [row[-1] for row in g].count(c) / float(l)
            #print('prop: %s' % prop)
            gini += (prop * (1.0 - prop))
    
    return gini

print(gini_index([[[1, 1], [1, 0]], [[1, 1], [1, 0]]], [0, 1]))
print(gini_index([[[1, 0], [1, 0]], [[1, 1], [1, 1]]], [0, 1]))

# first the score for the worst case at 1.0 followed by the score for the best case at 0.0.

# find the best split points
# A split is comprised of an attribute in the dataset and a value. We can summarize this as the index of an attribute to split and the value by which to split rows on that attribute. 
def test_split(index, value, dataset):
    
    left, right = list(), list()

    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    
    return left, right

def get_split(dataset):

    class_values = list(set(row[-1] for row in dataset))

    b_index, b_value, b_score, b_group = 999, 999, 999, None

    for index in range(len(dataset[0]) - 1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            #print(groups)
            #print('-'*10)
            gini = gini_index(groups, class_values)
            #print(gini)
            #print('**'*10)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return { 'index':b_index, 'value':b_value, 'groups':b_groups }

dataset = [[2.771244718,1.784783929,0],
    [1.728571309,1.169761413,0],
    [3.678319846,2.81281357,0],
    [3.961043357,2.61995032,0],
    [2.999208922,2.209014212,0],
    [7.497545867,3.162953546,1],
    [9.00220326,3.339047188,1],
    [7.444542326,0.476683375,1],
    [10.12493903,3.234550982,1],
    [6.642287351,3.319983761,1]]

split = get_split(dataset)
print(split)
print('Split: [X%d < %.3f]' % ((split['index']+1), split['value']))

# Build a Tree
def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)

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

def build_tree(train, max_depth, min_size):
    root = get_split(train)
    split(root, max_depth, min_size, 1)
    return root

def print_tree(node, depth = 0):
    if isinstance(node, dict):
        print('%s[X%d < %.3f]' % ((depth*'  ', (node['index'] + 1), node['value'])))
        print_tree(node['left'], depth + 1)
        print_tree(node['right'], depth + 1)
    else:
        print('%s[%s]' % ((depth*'  ', node)))

tree = build_tree(dataset, 3, 1)

print(tree)
print_tree(tree)


x = [[2.771244718, 1.784783929, 0], [1.728571309, 1.169761413, 1], [3.678319846, 2.81281357, 0], [3.961043357, 2.61995032, 1]]

def to_terminal_research(group):
    outcomes = [row[-1] for row in group]
    print(outcomes)
    print(outcomes.count)
    print(set(outcomes))
    return max(set(outcomes), key=outcomes.count)

print(to_terminal_research(x))