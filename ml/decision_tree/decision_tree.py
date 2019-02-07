import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

import math
from ml.math_tools import mt


class Node():
    def __init__(self, feature_i=None, threshold=None,
                 value=None, true_branch=None, false_branch=None):
        self.feature_i = feature_i  # Index for the feature that is tested
        self.threshold = threshold  # Threshold value for feature
        self.value = value  # Value if the node is a leaf in the tree
        self.true_branch = true_branch  # 'Left' subtree
        self.false_branch = false_branch  # 'Right' subtree


class DT():
    def __init__(self, min_samples_split=2, min_impurity=1e-7,
                 max_depth=float("inf"), loss=None):
        self.root = None  # Root node in dec. tree
        # Minimum n of samples to justify split
        self.min_samples_split = min_samples_split
        # The minimum impurity to justify split
        self.min_impurity = min_impurity
        # The maximum depth to grow the tree to
        self.max_depth = max_depth
        # Function to calculate impurity (classif.=>info gain, regr=>variance reduct.)
        # 切割树的方法，gini，方差等
        self._impurity_calculation = None
        # Function to determine prediction of y at leaf
        # 树节点取值的方法，分类树：选取出现最多次数的值，回归树：取所有值的平均值
        self._leaf_value_calculation = None
        # If y is one-hot encoded (multi-dim) or not (one-dim)
        self.one_dim = None
        # If Gradient Boost
        self.loss = loss

    def fit(self, X, y, loss=None):
        """
        Build decision tree
        """
        self.one_dim = len(np.shape(y)) == 1
        self.root = self._build_tree(X, y)
        self.loss = None

    def _build_tree(self, X, y, current_depth=0):
        """
        Recursive method which builds out the decision tree and splits X and respective y
        on the feature of X which (based on impurity) best separates the data
        """
        largest_impurity = 0
        best_criteria = None  # Feature index and threshold
        best_sets = None  # Subsets of the data

        # Check if expansion of y is needed
        if len(np.shape(y)) == 1:
            y = np.expand_dims(y, axis=1)

        """
        train data
        """
        Xy = np.concatenate((X, y), axis=1)

        """
        returns the shape of X as tuple
        n_samples: rows
        n_features: columns
        """
        n_samples, n_features = np.shape(X)

        """
        run when:
        1. n_samples >= 2
        2. current_depth <= float("inf"), unbounded upper value
        """
        if n_samples >= self.min_samples_split and current_depth <= self.max_depth:
            # Calculate the impurity for each feature
            for feature_i in range(n_features):
                # All values of feature_i
                feature_values = np.expand_dims(X[:, feature_i], axis=1)
                unique_values = np.unique(feature_values)
                # Iterate through all unique values of feature column i and
                # calculate the impurity
                for threshold in unique_values:
                    # Divide X and y depending on if the feature value of X at index feature_i
                    # meets the threshold
                    Xy1, Xy2 = mt.divide_on_feature(Xy, feature_i, threshold)

                    if len(Xy1) > 0 and len(Xy2) > 0:
                        # Select the y-values of the two sets
                        y1 = Xy1[:, n_features:]
                        y2 = Xy2[:, n_features:]

                        """
                        Calculate impurity
                        if classification tree, use information gain
                        if regression tree, use variance reduction
                        """
                        impurity = self._impurity_calculation(y, y1, y2)

                        """
                        If this threshold information gain bigger than previous
                        save the threshold value and the feature index
                        """
                        if impurity > largest_impurity:
                            largest_impurity = impurity
                            best_criteria = {
                                "feature_i": feature_i, "threshold": threshold}
                            best_sets = {
                                # X of left subtree
                                "leftX": Xy1[:, :n_features],
                                # y of left subtree
                                "lefty": Xy1[:, n_features:],
                                # X of right subtree
                                "rightX": Xy2[:, :n_features],
                                # y of right subtree
                                "righty": Xy2[:, n_features:]
                            }

        if largest_impurity > self.min_impurity:
            # Build subtrees for the right and left branches
            true_branch = self._build_tree(
                best_sets["leftX"], best_sets["lefty"], current_depth + 1)
            false_branch = self._build_tree(
                best_sets["rightX"], best_sets["righty"], current_depth + 1)
            return Node(feature_i=best_criteria["feature_i"], threshold=best_criteria[
                "threshold"], true_branch=true_branch, false_branch=false_branch)

        # We're at leaf => determine value
        leaf_value = self._leaf_value_calculation(y)
        return Node(value=leaf_value)

    def predict_value(self, x, tree=None):
        """
        Do a recursive search down the tree and make a prediction of the data sample by the
            value of the leaf that we end up at
        """

        if tree is None:
            tree = self.root

        # If we have a value (i.e we're at a leaf) => return value as the prediction
        if tree.value is not None:
            return tree.value

        # Choose the feature that we will test
        feature_value = x[tree.feature_i]

        # Determine if we will follow left or right branch
        branch = tree.false_branch
        if isinstance(feature_value, int) or isinstance(feature_value, float):
            if feature_value >= tree.threshold:
                branch = tree.true_branch
        elif feature_value == tree.threshold:
            branch = tree.true_branch

        # Test subtree
        return self.predict_value(x, branch)

    def predict(self, X):
        """
        Classify samples one by one and return the set of labels
        """
        y_pred = []
        for x in X:
            y_pred.append(self.predict_value(x))
        return y_pred

    def print_tree(self, tree=None, indent=" "):
        """
        Recursively print the decision tree
        """
        if not tree:
            tree = self.root

        # If we're at leaf => print the label
        if tree.value is not None:
            print(tree.value)
        # Go deeper down the tree
        else:
            # Print test
            print("%s:%s? " % (tree.feature_i, tree.threshold))
            # Print the true scenario
            print("%sT->" % (indent), end="")
            self.print_tree(tree.true_branch, indent + indent)
            # Print the false scenario
            print("%sF->" % (indent), end="")
            self.print_tree(tree.false_branch, indent + indent)


class CT(DT):
    """
    Classification Tree
    """

    def _calculate_information_gain(self, y, y1, y2):
        """
        切割树的标准，这里使用的是交叉熵
        """
        p = len(y1) / len(y)
        entropy = mt.calculate_entropy(y)
        info_gain = entropy - p * \
            mt.calculate_entropy(y1) - (1 - p) * \
            mt.calculate_entropy(y2)
        return info_gain

    def _majority_vote(self, y):
        """
        计算子节点值的方法，这里使用的是选取数据集中出现最多的种类
        """
        most_common = None
        max_count = 0
        for label in np.unique(y):
            # Count number of occurences of samples with label
            count = len(y[y == label])
            if count > max_count:
                most_common = label
                max_count = count
        return most_common

    def fit(self, X, y):
        self._impurity_calculation = self._calculate_information_gain
        self._leaf_value_calculation = self._majority_vote
        super(CT, self).fit(X, y)


class RT(DT):
    """
    Regression Tree
    """

    def _calculate_variance_reduction(self, y, y1, y2):
        """
        切割树的标准，这里使用的是平方残差
        """
        var_tot = mt.calculate_variance(y)
        var_1 = mt.calculate_variance(y1)
        var_2 = mt.calculate_variance(y2)
        frac_1 = len(y1) / len(y)
        frac_2 = len(y2) / len(y)

        # Calculate the variance reduction
        variance_reduction = var_tot - (frac_1 * var_1 + frac_2 * var_2)

        return sum(variance_reduction)

    def _mean_of_y(self, y):
        """
        计算子节点值的方法，这里使用的是取数据集中的平均值
        """
        value = np.mean(y, axis=0)
        return value if len(value) > 1 else value[0]

    def fit(self, X, y):
        self._impurity_calculation = self._calculate_variance_reduction
        self._leaf_value_calculation = self._mean_of_y
        super(RT, self).fit(X, y)
