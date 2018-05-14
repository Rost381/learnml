from random import randrange

def subsample(dataset, ratio):
    sample = list()
    n_sample = round(len(dataset) * ratio)
    while len(sample) < n_sample:
        index = randrange(len(dataset))
        sample.append(dataset[index])
    return sample

def random_forest(train, test, max_depth, min_size, sample_size, n_trees, n_features):
    trees = list()
    for i in range(n_trees):
        sample = subsample(train, sample_size)

        tree = build_tree(sample, max_depth, min_size, n_features)
        trees.append(tree)

    predictions = [bagging_predict(tree, row) for row in test]
    return(predictions)
