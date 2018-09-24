#!/usr/local/bin/python
# -*- coding: utf-8 -*-
"""
@Time: 2018/8/4 下午5:40
@Author: zhaoxingfeng
@Function：CART回归树
@Version: V1.2
"""
import numpy as np
import copy
import random


class Tree(object):
    def __init__(self):
        self.split_feature = None
        self.split_value = None
        self.split_gain = None
        self.internal_value = None
        self.node_index = None
        self.leaf_value = None
        self.tree_left = None
        self.tree_right = None

    def calc_predict_value(self, dataset):
        if self.leaf_value is not None:
            return self.leaf_value
        elif dataset[self.split_feature] <= self.split_value:
            return self.tree_left.calc_predict_value(dataset)
        else:
            return self.tree_right.calc_predict_value(dataset)

    # print tree structure by JSON
    def describe_tree(self):
        if not self.tree_left and not self.tree_right:
            leaf_info = "{leaf_value:" + str(self.leaf_value) + "}"
            return leaf_info
        left_info = self.tree_left.describe_tree()
        right_info = self.tree_right.describe_tree()
        tree_structure = "{split_feature:" + str(self.split_feature) + \
                         ",split_value:" + str(self.split_value) + \
                         ",split_gain:" + str(self.split_gain) + \
                         ",internal_value:" + str(self.internal_value) + \
                         ",node_index:" + str(self.node_index) + \
                         ",left_tree:" + left_info + \
                         ",right_tree:" + right_info + "}"
        return tree_structure

    # count all leaf nodes & parent nodes which have two leaf nodes
    def state_tree(self, leaves_state, node_state):
        if not self.tree_left and not self.tree_right:
            leaves_state.append(1)
            return
        if not self.tree_left.split_gain and not self.tree_right.split_gain:
            node_state.append([self.node_index, self.split_gain])
        self.tree_left.state_tree(leaves_state, node_state)
        self.tree_right.state_tree(leaves_state, node_state)
        return leaves_state, node_state

    # prune tree with given node_index
    def prune_tree(self, prune_node_index):
        if not self.tree_left and not self.tree_right:
            return
        if self.tree_left.node_index == prune_node_index:
            leaf_value = self.tree_left.internal_value
            self.tree_left = Tree()
            self.tree_left.node_index = prune_node_index
            self.tree_left.leaf_value = leaf_value
            return
        elif self.tree_right.node_index == prune_node_index:
            leaf_value = self.tree_right.internal_value
            self.tree_right = Tree()
            self.tree_right.node_index = prune_node_index
            self.tree_right.leaf_value = leaf_value
            return
        self.tree_left.prune_tree(prune_node_index)
        self.tree_right.prune_tree(prune_node_index)
        return


class BaseDecisionTree(object):
    def __init__(self, max_depth, num_leaves, min_samples_split, min_samples_leaf, subsample,
                 colsample_bytree, max_bin, min_child_weight, reg_gamma, reg_lambda, random_state):
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.max_bin = max_bin
        self.min_child_weight = min_child_weight
        self.reg_gamma = reg_gamma
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.tree = Tree()
        self.pred = None
        self.node_index = 0
        self.feature_importances_ = dict()

    def fit(self, dataset, targets):
        dataset_copy = copy.deepcopy(dataset).reset_index(drop=True)
        targets_copy = copy.deepcopy(targets).reset_index(drop=True)

        if self.random_state:
            random.seed(self.random_state)
        if self.subsample < 1.0:
            subset_index = random.sample(range(len(targets)), int(self.subsample*len(targets)))
            dataset_copy = dataset_copy.iloc[subset_index, :].reset_index(drop=True)
            targets_copy = targets_copy.iloc[subset_index, :].reset_index(drop=True)
        if self.colsample_bytree < 1.0:
            subcol_index = random.sample(dataset_copy.columns, int(self.colsample_bytree*len(dataset_copy.columns)))
            dataset_copy = dataset_copy[subcol_index]

        self.tree = self._fit(dataset_copy, targets_copy, depth=0)
        self.pred = dataset.apply(lambda x: self.predict(x), axis=1)

        leaves_state, node_state = self.tree.state_tree(leaves_state=[], node_state=[])
        while sum(leaves_state) > self.num_leaves:
            node_state = sorted(node_state, key=lambda x: x[1])
            self.tree.prune_tree(node_state[0][0])
            leaves_state, node_state = self.tree.state_tree(leaves_state=[], node_state=[])
        return self

    def _fit(self, dataset, targets, depth):
        if dataset.__len__() <= self.min_samples_split or targets['hess'].sum() <= self.min_child_weight:
            tree = Tree()
            tree.leaf_value = self.calc_leaf_value(targets)
            return tree

        if depth < self.max_depth:
            best_split_feature, best_split_value, best_split_gain, best_internal_value = \
                self.choose_best_feature(dataset, targets)
            left_dataset, right_dataset, left_targets, right_targets = \
                self.split_dataset(dataset, targets, best_split_feature, best_split_value)

            tree = Tree()
            if left_dataset.__len__() <= self.min_samples_leaf or \
                    right_dataset.__len__() <= self.min_samples_leaf:
                tree.leaf_value = self.calc_leaf_value(targets)
                return tree
            else:
                self.feature_importances_[best_split_feature] = \
                    self.feature_importances_.get(best_split_feature, 0) + 1

                tree.split_feature = best_split_feature
                tree.split_value = best_split_value
                tree.split_gain = best_split_gain
                tree.internal_value = best_internal_value
                tree.node_index = self.node_index
                self.node_index += 1
                tree.tree_left = self._fit(left_dataset, left_targets, depth+1)
                tree.tree_right = self._fit(right_dataset, right_targets, depth+1)
                return tree
        else:
            tree = Tree()
            tree.leaf_value = self.calc_leaf_value(targets)
            return tree

    def choose_best_feature(self, dataset, targets):
        best_split_gain = float('-inf')
        best_split_feature = None
        best_split_value = None

        for feature in dataset.columns:
            if dataset[feature].unique().__len__() <= 100:
                unique_values = dataset[feature].unique()
            else:
                unique_values = np.unique([np.percentile(dataset[feature], x)
                                           for x in np.linspace(0, 100, self.max_bin)])

            for split_value in unique_values:
                left_targets = targets[dataset[feature] <= split_value]
                right_targets = targets[dataset[feature] > split_value]
                split_gain = self.calc_split_gain(left_targets, right_targets)

                if split_gain > best_split_gain:
                    best_split_feature = feature
                    best_split_value = split_value
                    best_split_gain = split_gain
        best_internal_value = self.calc_leaf_value(targets)
        return best_split_feature, best_split_value, best_split_gain, best_internal_value

    def calc_leaf_value(self, targets):
        leaf_value = - targets['grad'].sum() / (targets['hess'].sum() + self.reg_lambda)
        return leaf_value

    def calc_split_gain(self, left_targets, right_targets):
        left_grad = left_targets['grad'].sum()
        left_hess = left_targets['hess'].sum()
        right_grad = right_targets['grad'].sum()
        right_hess = right_targets['hess'].sum()
        split_gain = 0.5 * (left_grad ** 2 / (left_hess + self.reg_lambda) +
                            right_grad ** 2 / (right_hess + self.reg_lambda) -
                            (left_grad + right_grad) ** 2 / (left_hess + right_hess + self.reg_lambda)) - self.reg_gamma
        return split_gain

    @staticmethod
    def split_dataset(dataset, targets, split_feature, split_value):
        left_dataset = dataset[dataset[split_feature] <= split_value]
        left_targets = targets[dataset[split_feature] <= split_value]
        right_dataset = dataset[dataset[split_feature] > split_value]
        right_targets = targets[dataset[split_feature] > split_value]
        return left_dataset, right_dataset, left_targets, right_targets

    def predict(self, dataset):
        return self.tree.calc_predict_value(dataset)

    def print_tree(self):
        return self.tree.describe_tree()
