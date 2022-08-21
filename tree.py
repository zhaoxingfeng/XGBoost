#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Time: 2018/8/4 下午5:40
@Author: zhaoxingfeng
@Function: CART决策树
"""
import numpy as np
import copy
import random


class Tree(object):
    """定义一棵cart决策树"""
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
        """
        模型打分

        :param dataset: pd.Series，一条样本
        :return: float，cart树预测值
        """
        if self.leaf_value is not None:
            return self.leaf_value
        
        if dataset[self.split_feature] <= self.split_value:
            return self.tree_left.calc_predict_value(dataset)
        else:
            return self.tree_right.calc_predict_value(dataset)

    def describe_tree(self):
        """
        以字典形式打印树结构
        
        :return: dict，树结构
        """
        if not self.tree_left and not self.tree_right:
            return {
                "leaf_value": self.leaf_value,
                "node_index": self.node_index
            }

        tree_left = self.tree_left.describe_tree()
        tree_right = self.tree_right.describe_tree()
        
        tree_structure = {
            "split_feature": self.split_feature,
            "split_value": self.split_value,
            "split_gain": self.split_gain,
            "internal_value": self.internal_value,
            "node_index": self.node_index,
            "tree_left": tree_left,
            "tree_right": tree_right
        }
        
        return tree_structure

    def state_tree(self, leaves_state, node_state):
        """
        统计叶子结点数量、非叶子结点编号和分裂增益
        
        :param leaves_state: list，存放叶子结点
        :param node_state: list，存在非叶子结点
        :return: str，树结构
        """
        # 叶子结点，+1
        if not self.tree_left and not self.tree_right:
            leaves_state.append(1)
            return

        # 非叶子结点，+[结点编号，分裂增益]
        if not self.tree_left.split_gain and not self.tree_right.split_gain:
            node_state.append([self.node_index, self.split_gain])
        
        # 递归向下统计
        self.tree_left.state_tree(leaves_state, node_state)
        self.tree_right.state_tree(leaves_state, node_state)

    def prune_tree(self, prune_node_index):
        """
        剪枝，给定父节点编号，剪掉左右叶子
        
        :param prune_node_index: int，待剪枝结点编号
        :return: None
        """
        if not self.tree_left and not self.tree_right:
            return
        
        # 从上向下递归寻找待剪枝结点，找到则删除左右结点
        if self.tree_left.node_index == prune_node_index:
            leaf_value = self.tree_left.internal_value
            self.tree_left = Tree()
            self.tree_left.node_index = prune_node_index
            self.tree_left.leaf_value = leaf_value
            return
        
        if self.tree_right.node_index == prune_node_index:
            leaf_value = self.tree_right.internal_value
            self.tree_right = Tree()
            self.tree_right.node_index = prune_node_index
            self.tree_right.leaf_value = leaf_value
            return
        
        # 该层没有找到，则继续寻找左右结点
        self.tree_left.prune_tree(prune_node_index)
        self.tree_right.prune_tree(prune_node_index)


class BaseDecisionTree(object):
    """cart决策树"""
    def __init__(self, max_depth, num_leaves, min_samples_split, min_samples_leaf, subsample,
                 colsample_bytree, max_bin, min_child_weight, reg_gamma, reg_lambda, random_state):
        """
        :param max_depth: int, cart树深度，-1表示不限制
        :param num_leaves: int, cart树最多叶子结点数量，-1表示不限制
        :param min_samples_split: int, cart树结点分裂所需的最小样本数
        :param min_samples_leaf: int, cart树叶子结点所需的最小样本数
        :param subsample: float, 样本行采样比例
        :param colsample_bytree: float, 样本列采样比例
        :param max_bin: int, 特征最大分桶数量，当特征取值过多（>max_bin）时生效
        :param min_child_weight: float,  cart树结点分裂所需的最小分裂增益(hessian)
        :param reg_gamma: float, L1正则
        :param reg_lambda: float, L2正则
        :param random_state: int/None, 随机数种子
        """
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
        # 记录结点编号
        self.node_index = 0
        self.feature_importances_ = dict()

    def fit(self, dataset, targets):
        """
        cart树训练入口

        :param dataset: pd.DataFrame，特征
        :param targets: pd.DataFrame，标签
        :return: self
        """
        dataset_copy = copy.deepcopy(dataset).reset_index(drop=True)
        targets_copy = copy.deepcopy(targets).reset_index(drop=True)

        # 样本行、列采样
        if self.random_state:
            random.seed(self.random_state)
        if self.subsample < 1.0:
            subset_index = random.sample(range(len(targets)), int(self.subsample * len(targets)))
            dataset_copy = dataset_copy.iloc[subset_index, :].reset_index(drop=True)
            targets_copy = targets_copy.iloc[subset_index, :].reset_index(drop=True)
        if self.colsample_bytree < 1.0:
            subcol_index = random.sample(dataset_copy.columns.tolist(), \
                int(self.colsample_bytree * len(dataset_copy.columns)))
            dataset_copy = dataset_copy[subcol_index]

        self.tree = self._fit(dataset_copy, targets_copy, depth=0)
        self.pred = dataset.apply(lambda x: self.predict(x), axis=1)

        # 剪枝，统计叶子结点数量，如果比预设值多，剪掉分裂增益小的叶子
        leaves_state, node_state = [], []
        self.tree.state_tree(leaves_state, node_state)
        while sum(leaves_state) > self.num_leaves:
            node_state = sorted(node_state, key=lambda x: x[1])
            self.tree.prune_tree(node_state[0][0])
            
            leaves_state, node_state = [], []
            self.tree.state_tree(leaves_state, node_state)
        return self

    def _fit(self, dataset, targets, depth):
        """
        递归建立决策树

        :param dataset: pd.DataFrame，特征
        :param targets: pd.DataFrame，标签
        :param depth: int，树深度
        :return: Tree
        """
        tree = Tree()
        # 给所有父结点、叶子结点编号
        tree.node_index = self.node_index
        self.node_index += 1
        
        # 如果该节点的样本小于分裂所需最小样本数量，或者二阶导数之和小于最小weight，则终止分裂
        if len(dataset) <= self.min_samples_split or targets['hess'].sum() <= self.min_child_weight:
            tree.leaf_value = self.calc_leaf_value(targets)
            return tree

        if depth < self.max_depth:
            best_split_feature, best_split_value, best_split_gain, internal_value = \
                self.choose_best_feature(dataset, targets)
            left_dataset, right_dataset, left_targets, right_targets = \
                self.split_dataset(dataset, targets, best_split_feature, best_split_value)

            # 如果父节点分裂后，左叶子节点/右叶子节点样本小于设置的叶子节点最小样本数量，则该父节点终止分裂
            if len(left_dataset) <= self.min_samples_leaf or \
                    len(right_dataset) <= self.min_samples_leaf:
                tree.leaf_value = self.calc_leaf_value(targets)
                return tree
            else:
                # 如果分裂的时候用到该特征，则该特征的importance加1
                self.feature_importances_[best_split_feature] = \
                    self.feature_importances_.get(best_split_feature, 0) + 1

                tree.split_feature = best_split_feature
                tree.split_value = best_split_value
                tree.split_gain = best_split_gain
                tree.internal_value = internal_value
                tree.tree_left = self._fit(left_dataset, left_targets, depth + 1)
                tree.tree_right = self._fit(right_dataset, right_targets, depth + 1)
                return tree
        # 如果树的深度超过预设值，则终止分裂
        else:
            tree.leaf_value = self.calc_leaf_value(targets)
            return tree

    def choose_best_feature(self, dataset, targets):
        """
        寻找最好的数据集划分方式，找到最优分裂特征、分裂阈值、分裂增益
        
        :param dataset: pd.DataFrame，样本特征
        :param targets: pd.DataFrame，样本标签
        :return: list
        """
        best_split_gain = float('-inf')
        best_split_feature = None
        best_split_value = None

        for feature in dataset.columns:
            if len(dataset[feature].unique()) <= 100:
                unique_values = dataset[feature].unique()
            # 如果该维度特征取值太多，则选择100个百分位值作为待选分裂阈值
            else:
                unique_values = np.unique([np.percentile(dataset[feature], x)
                                           for x in np.linspace(0, 100, self.max_bin)])

            # 对可能的分裂阈值求分裂增益，选取增益最大的阈值
            for split_value in unique_values:
                left_targets = targets[dataset[feature] <= split_value]
                right_targets = targets[dataset[feature] > split_value]
                
                split_gain = self.calc_split_gain(left_targets, right_targets)
                if split_gain > best_split_gain:
                    best_split_feature, best_split_value, best_split_gain = \
                        feature, split_value, split_gain
        
        # 如果不分裂，计算叶子结点值
        internal_value = self.calc_leaf_value(targets)

        return [best_split_feature, best_split_value, best_split_gain, internal_value]

    def calc_leaf_value(self, targets):
        """
        计算叶子结点值

        :param targets: pd.DataFrame，样本标签
        :return: float
        """
        leaf_value = - targets['grad'].sum() / (targets['hess'].sum() + self.reg_lambda)

        return leaf_value

    def calc_split_gain(self, left_targets, right_targets):
        """
        计算分裂增益

        :param left_targets: pd.DataFrame，样本标签
        :param right_targets: pd.DataFrame，样本标签
        :return: float
        """
        grad_left, hess_left = left_targets['grad'].sum(), left_targets['hess'].sum()
        grad_right, hess_right = right_targets['grad'].sum(), right_targets['hess'].sum()

        g_left = grad_left ** 2 / (hess_left + self.reg_lambda)
        g_right = grad_right ** 2 / (hess_right + self.reg_lambda)
        g_root = (grad_left + grad_right) ** 2 / (hess_left + hess_right + self.reg_lambda)
        
        gain_split = 0.5 * (g_left + g_right - g_root) - self.reg_gamma
        
        return gain_split

    @staticmethod
    def split_dataset(dataset, targets, split_feature, split_value):
        """
        划分样本到左右叶子

        :param dataset: pd.DataFrame，样本特征
        :param targets: pd.DataFrame，样本标签
        :param split_feature: string，待分裂特征名
        :param split_value: int/float，待分裂特征值
        :return: list
        """
        left_dataset = dataset[dataset[split_feature] <= split_value]
        left_targets = targets[dataset[split_feature] <= split_value]
        
        right_dataset = dataset[dataset[split_feature] > split_value]
        right_targets = targets[dataset[split_feature] > split_value]
        
        return [left_dataset, right_dataset, left_targets, right_targets]

    def predict(self, dataset):
        """
        模型打分

        :param dataset: pd.Series，单条样本
        :return: float
        """
        return self.tree.calc_predict_value(dataset)

    def print_tree(self):
        """
        打印树结构
        
        :return: dict
        """
        return self.tree.describe_tree()
