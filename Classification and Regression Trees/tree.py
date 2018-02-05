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

#first the score for the worst case at 1.0 followed by the score for the best case at 0.0.

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
            print(groups)
            print('-'*10)
            gini = gini_index(groups, class_values)
            print(gini)
            print('**'*10)
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