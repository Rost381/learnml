from split import *

def tree_to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)

def tree_create(train, max_depth, min_size):
    root = split_get(train)
    split(root, max_depth, min_size, 1)
    return root

def tree_print(node, depth = 0):
    if isinstance(node, dict):
        print('%s[X%d < %.3f]' % ((depth*'  ', (node['index'] + 1), node['value'])))
        tree_print(node['left'], depth + 1)
        tree_print(node['right'], depth + 1)
    else:
        print('%s[%s]' % ((depth*'  ', node)))