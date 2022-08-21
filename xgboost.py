#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Time: 2018/8/4 下午5:36
@Author: zhaoxingfeng
@Function: 精简版xgboost+lightgbm，支持连续特征
参考文献：
[1] Friedman JH. Greedy Function Approximation-A Gradient Boosting Machine[J].The Annals of Statistics,2001,29(5),1189-1232.
[2] Tianqi Chen. XGBoost: A Scalable Tree Boosting System[D].KDD2016,2016.
[3] 人工智能邂逅量化投资. XGBoost入门系列第一讲[DB/OL].https://zhuanlan.zhihu.com/p/27816315.
[4] 红色石头Will. 简单的交叉熵损失函数，你真的懂了吗？[DB/OL].https://blog.csdn.net/red_stone1/article/details/80735068.
[5] zhpmatrix. groot[DB/OL].https://github.com/zhpmatrix/groot.
"""
import math
import random
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import metrics
from tree import BaseDecisionTree


class BaseLoss(object):
    """损失函数基类"""
    def __init__(self):
        pass

    def grad(self, targets):
        """求一阶导"""
        pass

    def hess(self, targets):
        """求二阶导"""
        pass


class CrossEntropyLoss(BaseLoss):
    """
    交叉墒损失函数，L = -[y*log(pred) + (1-y)*log(1-pred)], y={1, 0}
    """
    @staticmethod
    def calc_pred(targets):
        """sigmoid函数求输出值"""
        pred = 1.0 / (1.0 + math.exp(-targets['pred']))
        return pred

    def grad(self, targets):
        """求一阶导"""
        pred = self.calc_pred(targets)
        grad = (- targets['label'] / pred + (1 - targets['label']) / (1 - pred)) * targets['class_weight']
        return grad

    def hess(self, targets):
        """求二阶导"""
        pred = self.calc_pred(targets)
        hess = (targets['label'] / pred ** 2 + (1 - targets['label']) / (1 - pred) ** 2) * targets['class_weight']
        return hess


class LogisticLoss(BaseLoss):
    """
    LR损失函数，L = ln(1+exp(-y*pred)), y={-1, 1}
    """
    def grad(self, targets):
        """求一阶导"""
        grad = (- targets['label'] / (1 + math.exp(targets['label'] * targets['pred']))) * targets['class_weight']
        return grad

    def hess(self, targets):
        """求二阶导"""
        hess = (math.exp(targets['label'] * targets['pred']) / (
            1 + math.exp(targets['label'] * targets['pred'])) ** 2) * targets['class_weight']
        return hess


class XGBClassifier(object):
    """scikit-learn接口"""
    def __init__(self, n_estimators=100, max_depth=-1, num_leaves=-1, learning_rate=0.1, min_samples_split=2,
                 min_samples_leaf=1, subsample=1., colsample_bytree=1., max_bin=225, min_child_weight=1.,
                 reg_gamma=0., reg_lambda=0., is_unbalance=True, random_state=None):
        """
        :param n_estimators: int, cart树数量
        :param max_depth: int, cart树深度，-1表示不限制
        :param num_leaves: int, cart树最多叶子结点数量，-1表示不限制
        :param learning_rate: float, 学习率
        :param min_samples_split: int, cart树结点分裂所需的最小样本数
        :param min_samples_leaf: int, cart树叶子结点所需的最小样本数
        :param subsample: float, 样本行采样比例
        :param colsample_bytree: float, 样本列采样比例
        :param max_bin: int, 特征最大分桶数量，当特征取值过多（>max_bin）时生效
        :param min_child_weight: float,  cart树结点分裂所需的最小分裂增益(hessian)
        :param reg_gamma: float, L1正则
        :param reg_lambda: float, L2正则
        :param is_unbalance: bool, 是否非平衡样本
        :param random_state: int/None, 随机数种子
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
        """
        模型训练入口

        :param dataset: pd.DataFrame，样本特征
        :param targets: pd.Series，样本标签
        :return: None
        """
        assert len(targets.unique()) == 2, "There must be two class for targets!"

        # 样本标签编码为0/1
        targets = targets.to_frame(name='label')
        targets['label'] = preprocessing.LabelEncoder().fit_transform(targets['label'])

        # 设置随机数种子，用于行/列采样
        if self.random_state:
            random.seed(self.random_state)
        random_state_stages = random.sample(range(self.n_estimators), self.n_estimators)

        # 设置非平衡样本集时，需进行正负样本权重分配，权重用于计算损失函数梯度
        if self.is_unbalance:
            class_weight = dict(targets.groupby(['label']).size() / min(targets.groupby(['label']).size()))
            class_weight = dict(zip(list(class_weight.keys()), list(class_weight.values())[::-1]))
        else:
            class_weight = dict(zip(targets['label'].unique(), [1, 1]))

        # 初始预测值，取0.5或其他值也可以，对收敛速度影响不大
        self.init_score = math.log((1 + targets['label'].mean()) / (1 - targets['label'].mean()))

        # targets包含建立cart树、计算损失函数所需的全部信息
        targets['class_weight'] = targets['label'].map(class_weight)
        targets['pred'] = self.init_score
        targets['grad'] = targets.apply(self.loss.grad, axis=1)
        targets['hess'] = targets.apply(self.loss.hess, axis=1)

        # 建立多个cart决策树
        for stage in range(self.n_estimators):
            print(("iter: {0}".format(stage + 1).center(80, '=')))
            tree = BaseDecisionTree(self.max_depth, self.num_leaves, self.min_samples_split, self.min_samples_leaf,
                                    self.subsample, self.colsample_bytree, self.max_bin, self.min_child_weight,
                                    self.reg_gamma, self.reg_lambda, random_state_stages[stage])
            tree.fit(dataset, targets)
            self.trees[stage] = tree
            print(tree.print_tree())

            # 每一轮建树结束，更新预测值和梯度
            targets['pred'] = targets['pred'] + self.learning_rate * tree.pred
            targets['grad'] = targets.apply(self.loss.grad, axis=1)
            targets['hess'] = targets.apply(self.loss.hess, axis=1)

            # 计算特征重要性，（1）特征分裂次数（2）特征分裂增益，这里采用前者
            for key, value in tree.feature_importances_.items():
                self.feature_importances_[key] = self.feature_importances_.get(key, 0) + 1

    def predict_proba(self, dataset):
        """
        模型打分p_value

        :param dataset: pd.DataFrame，样本特征
        :return: np.array，[p(label=0), p(label=1)]
        """
        res = []
        for index, row in dataset.iterrows():
            score = self.init_score
            for stage, tree in self.trees.items():
                score += self.learning_rate * tree.predict(row)
            p_1 = 1.0 / (1 + math.exp(-score))
            res.append([1 - p_1, p_1])
        return np.array(res)

    def predict(self, dataset):
        """
        根据模型打分p_value，以0.5作为阈值进行label映射，>=0.5为正反之为负

        :param dataset: pd.DataFrame，样本特征
        :return: np.array，[label]
        """
        res = []
        for p in self.predict_proba(dataset):
            label = 0 if p[0] >= p[1] else 1
            res.append(label)
        return np.array(res)


if __name__ == '__main__':
    df = pd.read_csv("source/pima_indians.csv")
    xgb = XGBClassifier(n_estimators=5,
                        max_depth=6,
                        num_leaves=10,
                        learning_rate=0.1,
                        min_samples_split=40,
                        min_samples_leaf=5,
                        subsample=0.6,
                        colsample_bytree=0.8,
                        max_bin=150,
                        min_child_weight=1,
                        reg_gamma=0.1,
                        reg_lambda=0.3,
                        is_unbalance=False,
                        random_state=66)
    
    df_train = df.loc[:int(0.7 * len(df)), :]
    df_test = df.loc[int(0.7 * len(df)):, :]
    
    features = ["Number", "Plasma", "Diastolic", "Triceps", "2-Hour", "Body", "Diabetes", "Age"]
    xgb.fit(df_train[features], df_train['Class'])
    
    print(metrics.roc_auc_score(df_train["Class"], xgb.predict_proba(df_train[features])[:, 1]))
    print(metrics.roc_auc_score(df_test["Class"], xgb.predict_proba(df_test[features])[:, 1]))
