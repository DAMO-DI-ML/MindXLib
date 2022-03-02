from typing import List, Tuple
import numpy as np
import pandas as pd
from mlxtend import frequent_patterns as mlx
from mindxlib.utils import features
from sklearn.ensemble import RandomForestClassifier as RF


def extract_fpgrowth(X,
                     minsupport=0.1,
                     maxcardinality=2,
                     verbose=False) -> List[Tuple]:
    """
    extract frequent itemsets as rules from a one-hot dataframe

    :param
        X: pandas DataFrame, shape (n_samples, n_features), with onehot encoded format 0/1.
    :param
        minsupport: float, between 0 and 1 for minimum support of the itemsets returned.
                    The support is computed as the fraction transactions_where_item(s)_occur / total_transactions.
    :param
        maxcardinality: Maximum length of the itemsets generated. If `None` (default) all
                        possible itemsets lengths are evaluated.
    :param verbose:
    :return: list of itemsets
    """

    itemsets_df = mlx.fpgrowth(X, min_support=minsupport, max_len=maxcardinality)
    itemsets_indices = [tuple(s[1]) for s in itemsets_df.values]
    itemsets = [np.array(X.columns)[list(inds)] for inds in itemsets_indices]
    itemsets = list(map(tuple, itemsets))
    if verbose:
        print(len(itemsets), 'rules mined')

    return itemsets


def extract_rules_from_tree(clf, X):  # 单棵树提取rule,仅输出正类对应的rule
    """
    extract rule from a single tree, which is already trained
    :param clf: tree estimator
    :param X: pandas DataFrame, shape (n_samples, n_features)
    :return: list of rules
    """
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold

    def find_path(node_numb, path, x):
        path.append(node_numb)
        if node_numb == x:
            return True
        left = False
        right = False
        if children_left[node_numb] != -1:
            left = find_path(children_left[node_numb], path, x)
        if children_right[node_numb] != -1:
            right = find_path(children_right[node_numb], path, x)
        if left or right:
            return True
        path.remove(node_numb)
        return False

    def get_rule(path, column_names):
        path_rules = []
        single_rule = []
        for index, node in enumerate(path):
            if index != len(path) - 1:
                if children_left[node] == path[index + 1]:
                    single_rule.append("{}<={}".format(column_names[feature[node]], threshold[node]))
                else:
                    single_rule.append("{}>{}".format(column_names[feature[node]], threshold[node]))
        for i in range(1,len(single_rule)):
            tmp_rule = single_rule[0:i]
            # print(tmp_rule)
            path_rules.append(tmp_rule)
        return path_rules

    # Leaves
    leave_id = clf.apply(X)
    # print(leave_id)
    paths = {}
    for leaf in np.unique(leave_id):
        if clf.classes_[np.argmax(clf.tree_.value[leaf])] == 1: #仅关注正类输出的rule
            path_leaf = []
            find_path(0, path_leaf, leaf)
            # print(path_leaf)
            paths[leaf] = np.unique(np.sort(path_leaf))

    rules = []
    for key in paths:
        # print(get_rule(paths[key], data_df.columns))
        rules += get_rule(paths[key], X.columns)
    # print(rules)
    return rules


def extract_rf(X, y, n_trees=100):
    """
    extract rules from random forest
    :param X: pandas DataFrame, shape (n_samples, n_features)
    :param y: shape (n_samples,)
             The target values (class labels in classification, real numbers in
             regression).
    :param n_trees: int, default=100   The number of trees in the forest.
    :return:
    """
    model = RF(n_estimators=n_trees)
    model.fit(X, y)

    all_rules = []
    for clf in model.estimators_:
        single_tree_rule_list = extract_rules_from_tree(clf, X)
        all_rules += single_tree_rule_list
    return all_rules


