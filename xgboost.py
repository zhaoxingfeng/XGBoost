# -*- coding: utf-8 -*-
"""
@Time: 2018/8/4 下午5:36
@Author: zhaoxingfeng
@Function：xgboost二分类，连续特征
@Version: V1.0
参考文献：
[1] Tianqi Chen. XGBoost: A Scalable Tree Boosting System[D].KDD2016,2016.
[2] 人工智能邂逅量化投资. XGBoost入门系列第一讲[DB/OL].https://zhuanlan.zhihu.com/p/27816315.
[3] zhpmatrix. groot[DB/OL].https://github.com/zhpmatrix/groot.
"""
from __future__ import division
import pandas as pd
import numpy as np
from math import exp, log
from tree import BaseDecisionTree
import warnings
warnings.filterwarnings('ignore')
pd.set_option('precision', 4)
pd.set_option('display.max_rows', 50)
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('expand_frame_repr', False)


class LogisticLoss(object):
    @staticmethod
    def calc_grad(targets):
        preds = [1.0 / (1.0 + exp(-pred)) for pred in targets['pred']]
        grad = [- label / pred + (1 - label) / (1 - pred) for (label, pred) in zip(targets['label'], preds)]
        return grad

    @staticmethod
    def calc_hess(targets):
        preds = [1.0 / (1.0 + exp(-pred)) for pred in targets['pred']]
        hess = [label / pred**2 + (1 - label) / (1 - pred)**2 for (label, pred) in zip(targets['label'], preds)]
        return hess


class XGBClassifier(object):
    def __init__(self, n_estimators=100, max_depth=2**31-1, learning_rate=0.1, min_samples_split=2,
                 min_samples_leaf=1, subsample=1., colsample_bytree=1., max_bin=225, min_child_weight=1.,
                 reg_gamma=0., reg_lambda=0., loss="logistic", random_state=None):
        """Construct a xgboost model

        Parameters
        ----------
        n_estimators : int, optional (default=100)
            Number of boosted trees to fit.
        max_depth : int, optional (default=2**31-1)
            Maximum tree depth for base learners, -1 means no limit.
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
        random_state : int or None, optional (default=None)
            Random number seed.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.max_bin = max_bin
        self.min_child_weight = min_child_weight
        self.reg_gamma = reg_gamma
        self.reg_lambda = reg_lambda
        self.loss = loss
        self.random_state = random_state
        self.pred_0 = None
        self.trees = dict()
        self.feature_importance = dict()

    def fit(self, dataset, targets):
        if self.loss == 'logistic':
            self.loss = LogisticLoss()
        else:
            self.loss = LogisticLoss()

        targets = targets.to_frame(name='label')
        if targets['label'].unique().__len__() != 2:
            raise ValueError("There must be two class for targets!")
        if len([x for x in dataset.columns if dataset[x].dtype in ['int32', 'float32', 'int64', 'float64']]) \
                != len(dataset.columns):
            raise ValueError("The features dtype must be int or float!")

        mean = 1.0 * sum(targets['label']) / len(targets['label'])
        self.pred_0 = 0.5 * log((1 + mean) / (1 - mean))
        targets['pred'] = self.pred_0
        targets['grad'] = self.loss.calc_grad(targets)
        targets['hess'] = self.loss.calc_hess(targets)

        for stage in range(self.n_estimators):
            print(str(stage).center(80, '='))
            tree = BaseDecisionTree(self.max_depth, self.min_samples_split, self.min_samples_leaf, self.subsample,
                                    self.colsample_bytree, self.max_bin, self.min_child_weight, self.reg_gamma,
                                    self.reg_lambda, self.random_state)
            tree.fit(dataset, targets)
            self.trees[stage] = tree
            targets['pred'] = targets['pred'] + self.learning_rate * tree.pred
            targets['grad'] = self.loss.calc_grad(targets)
            targets['hess'] = self.loss.calc_hess(targets)

            for key, value in tree.feature_importance.items():
                self.feature_importance[key] = self.feature_importance.get(key, 0) + 1
        print(sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True))

    def predict_proba(self, dataset):
        res = []
        for index, row in dataset.iterrows():
            f_value = self.pred_0
            for stage, tree in self.trees.items():
                f_value += self.learning_rate * tree.predict(row)
            p_0 = 1.0 / (1 + exp(2 * f_value))
            res.append([p_0, 1 - p_0])
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
                        learning_rate=0.3,
                        min_samples_split=10,
                        min_samples_leaf=3,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        max_bin=225,
                        min_child_weight=0.1,
                        reg_gamma=0.0,
                        reg_lambda=1.0,
                        loss='logistic',
                        random_state=4)

    train_count = int(0.7 * len(df))
    xgb.fit(df.ix[:train_count, :-1], df.ix[:train_count, 'Class'])
    prob = xgb.predict_proba(df.ix[train_count:, :-1])[:, 1]

    from sklearn import metrics
    fpr, tpr, threshold = metrics.roc_curve(df.ix[train_count:, 'Class'], prob, pos_label=1)
    print(metrics.auc(fpr, tpr))
