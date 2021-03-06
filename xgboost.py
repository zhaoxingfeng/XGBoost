#!/usr/local/bin/python
# -*- coding: utf-8 -*-
"""
@Time: 2018/8/4 下午5:36
@Author: zhaoxingfeng
@Function：xgboost二分类，连续特征
@Version: V1.2
参考文献：
[1] Friedman JH. Greedy Function Approximation-A Gradient Boosting Machine[J].The Annals of Statistics,2001,29(5),1189-1232.
[2] Tianqi Chen. XGBoost: A Scalable Tree Boosting System[D].KDD2016,2016.
[3] 人工智能邂逅量化投资. XGBoost入门系列第一讲[DB/OL].https://zhuanlan.zhihu.com/p/27816315.
[4] 红色石头Will. 简单的交叉熵损失函数，你真的懂了吗？[DB/OL].https://blog.csdn.net/red_stone1/article/details/80735068.
[5] zhpmatrix. groot[DB/OL].https://github.com/zhpmatrix/groot.
"""
from __future__ import division
import pandas as pd
import numpy as np
from math import exp, log
from tree import BaseDecisionTree
import random
pd.set_option('precision', 4)
pd.set_option('display.max_rows', 50)
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('expand_frame_repr', False)
from sklearn import preprocessing


class BaseLoss(object):
    def __init__(self):
        pass

    def grad(self, targets):
        pass

    def hess(self, targets):
        pass


class CrossEntropyLoss(BaseLoss):
    """
    L = -[y*log(pred) + (1-y)*log(1-pred)], y={1, 0}
    """
    @staticmethod
    def calc_pred(targets):
        pred = 1.0 / (1.0 + exp(- targets['pred']))
        return pred

    def grad(self, targets):
        pred = self.calc_pred(targets)
        grad = - targets['label'] / pred + (1 - targets['label']) / (1 - pred) * targets['class_weight']
        return grad

    def hess(self, targets):
        pred = self.calc_pred(targets)
        hess = targets['label'] / pred**2 + (1 - targets['label']) / (1 - pred)**2 * targets['class_weight']
        return hess

# Not used.
class LogisticLoss(BaseLoss):
    """
    L = ln(1+exp(-y*pred)), y={-1, 1}
    """
    def grad(self, targets):
        grad = - targets['label'] / (1 + exp(targets['label'] * targets['pred'])) * targets['class_weight']
        return grad

    def hess(self, targets):
        hess = exp(targets['label'] * targets['pred']) / (1 + exp(targets['label'] * targets['pred']))**2 * targets['class_weight']
        return hess


class XGBClassifier(object):
    def __init__(self, n_estimators=100, max_depth=-1, num_leaves=-1, learning_rate=0.1, min_samples_split=2,
                 min_samples_leaf=1, subsample=1., colsample_bytree=1., max_bin=225, min_child_weight=1.,
                 reg_gamma=0., reg_lambda=0., is_unbalance=True, random_state=None):
        """Construct a xgboost model

        Parameters
        ----------
        n_estimators : int, optional (default=100)
            Number of boosted trees to fit.
        max_depth : int, optional (default=-1)
            Maximum tree depth for base learners, -1 means no limit.
        num_leaves : int, optional (default=-1)
            Maximum tree leaves for base learners, -1 means no limit.
        learning_rate : float, optional (default=0.1)
            Boosting learning rate.
        min_samples_split : int, optional (default=2)
            The minimum number of samples required to split an internal node.
        min_samples_leaf : int, optional (default=1)
            The minimum number of samples required to be at a leaf node.
        subsample : float, optional (default=1.)
            Subsample ratio of the training instance.
        colsample_bytree : float, optional (default=1.)
            Subsample ratio of columns when constructing each tree.
        max_bin: int or None, optional (default=225))
            Max number of discrete bins for features.
        min_child_weight : float, optional (default=1.)
            Minimum sum of instance weight(hessian) needed in a child(leaf).
        reg_gamma : float, optional (default=0.)
            L1 regularization term on weights.
        reg_lambda : float, optional (default=0.)
            L2 regularization term on weights.
        is_unbalance : bool, optional (default=True.)
            Set this to ``True`` if training data are unbalance.
        random_state : int or None, optional (default=None)
            Random number seed.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth if max_depth != -1 else float('inf')
        self.num_leaves = num_leaves if num_leaves != -1 else float('inf')
        self.learning_rate = learning_rate
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.max_bin = max_bin
        self.min_child_weight = min_child_weight
        self.reg_gamma = reg_gamma
        self.reg_lambda = reg_lambda
        self.is_unbalance = is_unbalance
        self.random_state = random_state
        self.init_score = None
        self.trees = dict()
        self.feature_importances_ = dict()
        self.loss = CrossEntropyLoss()

    def fit(self, dataset, targets):
        assert targets.unique().__len__() == 2, "There must be two class for targets!"
        targets = targets.to_frame(name='label')
        # transform labels into 0&1
        targets['label'] = preprocessing.LabelEncoder().fit_transform(targets['label'])

        if self.random_state:
            random.seed(self.random_state)
        random_state_stages = random.sample(range(self.n_estimators), self.n_estimators)

        if self.is_unbalance:
            class_weight = dict(targets.groupby(['label']).size() * 1.0 / min(targets.groupby(['label']).size()))
            class_weight = dict(zip(list(class_weight.keys()), list(class_weight.values())[::-1]))
        else:
            class_weight = dict(zip(targets['label'].unique(), [1, 1]))

        # base score
        self.init_score = log((1 + targets['label'].mean()) / (1 - targets['label'].mean()))
        targets['class_weight'] = targets['label'].map(class_weight)
        targets['pred'] = self.init_score
        targets['grad'] = targets.apply(self.loss.grad, axis=1)
        targets['hess'] = targets.apply(self.loss.hess, axis=1)

        for stage in range(self.n_estimators):
            print(("iter: "+str(stage+1)).center(80, '='))
            tree = BaseDecisionTree(self.max_depth, self.num_leaves, self.min_samples_split, self.min_samples_leaf,
                                    self.subsample, self.colsample_bytree, self.max_bin, self.min_child_weight,
                                    self.reg_gamma, self.reg_lambda, random_state_stages[stage])
            tree.fit(dataset, targets)
            self.trees[stage] = tree
            print(tree.print_tree())

            targets['pred'] = targets['pred'] + self.learning_rate * tree.pred
            targets['grad'] = targets.apply(self.loss.grad, axis=1)
            targets['hess'] = targets.apply(self.loss.hess, axis=1)

            for key, value in tree.feature_importances_.items():
                self.feature_importances_[key] = self.feature_importances_.get(key, 0) + 1

    def predict_proba(self, dataset):
        res = []
        for index, row in dataset.iterrows():
            score = self.init_score
            for stage, tree in self.trees.items():
                score += self.learning_rate * tree.predict(row)
            p_1 = 1.0 / (1 + exp(-score))
            res.append([1 - p_1, p_1])
        return np.array(res)

    def predict(self, dataset):
        res = []
        for p in self.predict_proba(dataset):
            label = 0 if p[0] >= p[1] else 1
            res.append(label)
        return np.array(res)


if __name__ == '__main__':
    df = pd.read_csv(r"source/pima indians.csv")
    xgb = XGBClassifier(n_estimators=5,
                        max_depth=6,
                        num_leaves=30,
                        learning_rate=0.1,
                        min_samples_split=40,
                        min_samples_leaf=10,
                        subsample=0.6,
                        colsample_bytree=0.8,
                        max_bin=150,
                        min_child_weight=1,
                        reg_gamma=0.1,
                        reg_lambda=0.3,
                        is_unbalance=False,
                        random_state=66)
    train_count = int(0.7 * len(df))
    xgb.fit(df.ix[:train_count, :-1], df.ix[:train_count, 'Class'])

    from sklearn import metrics
    print(metrics.roc_auc_score(df.ix[:train_count, 'Class'], xgb.predict_proba(df.ix[:train_count, :-1])[:, 1]))
    print(metrics.roc_auc_score(df.ix[train_count:, 'Class'], xgb.predict_proba(df.ix[train_count:, :-1])[:, 1]))
