import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

import math
from ml.math_tools import mt


class DT():
    def find_split(self, index, value, dataset):
        """
        find the best split point
        """
        left, right = list(), list()
        for row in dataset:
            if row[index] < value:
                left.append(row)
            else:
                right.append(row)
        return left, right

    def gini_index(self, groups, classes):
        """
        calculate Gini Index
        """
        # count all samples at split point
        n_instances = float(sum([len(group) for group in groups]))
        # sum weighted Gini index for each group
        gini = 0.0
        for group in groups:
            size = float(len(group))
            # avoid divide by zero
            if size == 0:
                continue
            score = 0.0
            # score the group based on the score for each class
            for class_val in classes:
                p = [row[-1] for row in group].count(class_val) / size
                score += p * p
            # weight the group score by its relative size
            gini += (1.0 - score) * (size / n_instances)
        return gini

    def get_split(self, dataset):
        """
        best split point for a dataset
        """
        class_values = list(set(row[-1] for row in dataset))
        b_index, b_value, b_score, b_groups = 999, 999, 999, None
        for index in range(len(dataset[0]) - 1):
            for row in dataset:
                groups = self.find_split(index, row[index], dataset)
                gini = self.gini_index(groups, class_values)
                if gini < b_score:
                    b_index, b_value, b_score, b_groups = index, row[index], gini, groups
        return {'index': b_index, 'value': b_value, 'groups': b_groups}

    def tree_to_terminal(self, group):
        """
        create terminal node value
        """
        outcomes = [row[-1] for row in group]
        return max(set(outcomes), key=outcomes.count)

    def split(self, node, max_depth, min_size, depth):
        """
        create child splits for a node or make terminal
        """
        left, right = node['groups']
        del(node['groups'])
        # check for a no split
        if not left or not right:
            node['left'] = node['right'] = self.tree_to_terminal(left + right)
            return
        # check for max depth
        if depth >= max_depth:
            node['left'], node['right'] = self.tree_to_terminal(
                left), self.tree_to_terminal(right)
            return
        # process left child
        if len(left) <= min_size:
            node['left'] = self.tree_to_terminal(left)
        else:
            node['left'] = self.get_split(left)
            self.split(node['left'], max_depth, min_size, depth + 1)
        # process right child
        if len(right) <= min_size:
            node['right'] = self.tree_to_terminal(right)
        else:
            node['right'] = self.get_split(right)
            self.split(node['right'], max_depth, min_size, depth + 1)

    def build_tree(self, train, max_depth, min_size):
        """
        build decision tree
        """
        root = self.get_split(train)
        self.split(root, max_depth, min_size, 1)
        return root

    def predict(self, node, row):
        """
        prdiction with decision tree
        """
        if row[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return self.predict(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self.predict(node['right'], row)
            else:
                return node['right']
