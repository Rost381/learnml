import numpy as np
import pandas as pd
import random

import math
from zero.utils.stats import calculate_entropy, calculate_variance


class Node():
    def __init__(self,
                 feature_i=None,
                 threshold=None,
                 value=None,
                 true_branch=None,
                 false_branch=None):
        self.feature_i = feature_i  # Index for the feature that is tested
        self.threshold = threshold  # Threshold value for feature
        self.value = value  # Value if the node is a leaf in the tree
        self.true_branch = true_branch  # 'Left' subtree
        self.false_branch = false_branch  # 'Right' subtree


class DecisionTree():
    """Base Tree model
    Super class of ClassificationTree and RegressionTree

    Parameters:
    -----------
    min_samples_split : int
        The minimum number of samples needed to make a split when building a tree.
    min_impurity_split : float
        The minimum impurity required to split tree.
    max_depth : int
        The maximum depth of a tree.
    """

    def __init__(self,
                 min_samples_split=2,
                 min_impurity_split=1e-7,
                 max_depth=float("inf")):
        self.root = None
        self.min_samples_split = min_samples_split
        self.min_impurity_split = min_impurity_split
        self.max_depth = max_depth

        """Function to calculate impurity
        ClassificationTree : info gain
            Information gain is used to decide which feature to
            split on at each step in building the tree.

        RegressionTree : variance reduction
            variance reduction is often employed in cases where
            the target variable is continuous (regression tree),
            meaning that use of many other metrics would
            first require discretization before being applied. 
        """
        self._impurity_calculation = None

        """Function to determine prediction of y at leaf
        ClassificationTree : max count
        RegressionTree : mean(y)
        """
        self._leaf_value_calculation = None

    def _divide_on_feature(self, X, feature_i, threshold):
        """Used for decision tree
        Divide dataset based on if sample value on 
        feature index is larger than the given threshold
        """
        split_func = None
        if isinstance(threshold, int) or isinstance(threshold, float):
            def split_func(sample): return sample[feature_i] >= threshold
        else:
            def split_func(sample): return sample[feature_i] == threshold

        X_1 = np.array([sample for sample in X if split_func(sample)])
        X_2 = np.array([sample for sample in X if not split_func(sample)])

        return np.array([X_1, X_2])

    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y, current_depth=0):
        """Recursive method which builds out the decision tree and splits X and
        respective y on the feature of X which (based on impurity) best separates the data
        """
        largest_impurity = 0
        best_criteria = None  # Feature index and threshold
        best_sets = None  # Subsets of the data

        # Check if expansion of y is needed
        if len(np.shape(y)) == 1:
            y = np.expand_dims(y, axis=1)

        """Xy : Train data"""
        Xy = np.concatenate((X, y), axis=1)

        """Returns the shape of X as tuple
        n_samples : rows
        n_features : columns
        """
        n_samples, n_features = np.shape(X)

        """Runs when:
        n_samples >= min_samples_split and
        current_depth <= max_depth
        """
        if n_samples >= self.min_samples_split and current_depth <= self.max_depth:
            """Calculate the impurity for each feature"""
            for feature_i in range(n_features):
                """All values of feature_i"""
                feature_values = np.expand_dims(X[:, feature_i], axis=1)
                unique_values = np.unique(feature_values)

                """Find best_criteria and best_sets"""
                for threshold in unique_values:
                    """Divide X and y depending on threshold
                    Xy: train dataset
                    feature_i : column order
                    threshold : unique_values in this column order

                    Xy_left : the value in row >= threshold in row
                    Xy_right : the value in row < threshold in row
                    """
                    Xy_left, Xy_right = self._divide_on_feature(
                        Xy, feature_i, threshold)

                    if len(Xy_left) > 0 and len(Xy_right) > 0:
                        """Select the y values of the two sets """
                        y1 = Xy_left[:, n_features:]
                        y2 = Xy_right[:, n_features:]

                        """Calculate impurity
                        | classification tree | information gain   |
                        --------------------------------------------
                        | regression tree     | variance reduction |
                        """
                        impurity = self._impurity_calculation(y, y1, y2)

                        """If this threshold information gain bigger than previous
                        save the threshold value and the feature index
                        """
                        if impurity > largest_impurity:
                            largest_impurity = impurity
                            best_criteria = {
                                "feature_i": feature_i, "threshold": threshold}
                            best_sets = {
                                # X of left subtree
                                "leftX": Xy_left[:, :n_features],
                                # y of left subtree
                                "lefty": Xy_left[:, n_features:],
                                # X of right subtree
                                "rightX": Xy_right[:, :n_features],
                                # y of right subtree
                                "righty": Xy_right[:, n_features:]
                            }
        """When loop finished, continue to build tree"""
        if largest_impurity > self.min_impurity_split:
            """Build subtrees for the right and left branches"""
            true_branch = self._build_tree(
                best_sets["leftX"], best_sets["lefty"], current_depth + 1)
            false_branch = self._build_tree(
                best_sets["rightX"], best_sets["righty"], current_depth + 1)
            return Node(feature_i=best_criteria["feature_i"],
                        threshold=best_criteria["threshold"],
                        true_branch=true_branch,
                        false_branch=false_branch)

        """We're at leaf now
        | classification tree | majority vote |
        ---------------------------------------
        | regression tree     | mean(y)       |
        """
        leaf_value = self._leaf_value_calculation(y)
        return Node(value=leaf_value)

    def predict_value(self, X, tree=None):
        """Recursive search down the tree and make a prediction of the data sample
        by the value of the leaf
        """
        if tree is None:
            tree = self.root

        """If we have a value (i.e we're at a leaf) => return value as the prediction"""
        if tree.value is not None:
            return tree.value

        """Choose the feature that we will test"""
        feature_value = X[tree.feature_i]

        """Determine if we will follow left or right branch"""
        branch = tree.false_branch
        if isinstance(feature_value, int) or isinstance(feature_value, float):
            if feature_value >= tree.threshold:
                branch = tree.true_branch
        elif feature_value == tree.threshold:
            branch = tree.true_branch

        """Test subtree"""
        return self.predict_value(X, branch)

    def predict(self, X):
        """Classify samples one by one and return the set of labels"""
        y_pred = []
        for x in X:
            y_pred.append(self.predict_value(x))
        return y_pred

    def print_tree(self, tree=None, indent=" "):
        """Recursively print the decision tree"""
        if not tree:
            tree = self.root

        """If we're at leaf => print the label"""
        if tree.value is not None:
            print(tree.value)
        # Go deeper down the tree
        else:
            # Print test
            print("%s:%s?" % (tree.feature_i, tree.threshold))
            # Print the true scenario
            print("%s True ->" % (indent), end="")
            self.print_tree(tree.true_branch, indent + indent)
            # Print the false scenario
            print("%s False ->" % (indent), end="")
            self.print_tree(tree.false_branch, indent + indent)


class ClassificationTree(DecisionTree):
    """Classification Tree"""

    def _info_gain(self, y, y1, y2):
        """Split tree by info gain"""
        p = len(y1) / len(y)
        entropy = calculate_entropy(y)
        info_gain = entropy - p * \
            calculate_entropy(y1) - (1 - p) * \
            calculate_entropy(y2)
        return info_gain

    def _majority_vote(self, y):
        """Leaf, max count"""
        most_common = None
        max_count = 0
        for label in np.unique(y):

            count = len(y[y == label])
            if count > max_count:
                most_common = label
                max_count = count
        return most_common

    def fit(self, X, y):
        self._impurity_calculation = self._info_gain
        self._leaf_value_calculation = self._majority_vote
        super(ClassificationTree, self).fit(X, y)


class RegressionTree(DecisionTree):
    """Regression Tree"""

    def _variance_reduction(self, y, y1, y2):
        """Split tree by variance reduction"""
        var_tot = calculate_variance(y)
        var_1 = calculate_variance(y1)
        var_2 = calculate_variance(y2)
        frac_1 = len(y1) / len(y)
        frac_2 = len(y2) / len(y)
        variance_reduction = var_tot - (frac_1 * var_1 + frac_2 * var_2)
        return sum(variance_reduction)

    def _mean(self, y):
        """Leaf, mean"""
        value = np.mean(y, axis=0)
        return value if len(value) > 1 else value[0]

    def fit(self, X, y):
        self._impurity_calculation = self._variance_reduction
        self._leaf_value_calculation = self._mean
        super(RegressionTree, self).fit(X, y)
