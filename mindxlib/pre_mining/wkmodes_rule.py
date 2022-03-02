# coding=utf-8
from multiprocessing import cpu_count
# import random
import sklearn
from dask import delayed
import dask.array as da
import pandas as pd
import numpy as np
from collections import defaultdict
import time
import math
from mindxlib.utils import features


def matching_dissim(a, b):
    # return np.sum(a != b, axis=1)
    return a.dot(1 - 2 * b) + np.sum(b)


def init_random(X, n_clusters):
    seeds = np.random.RandomState(42).permutation(len(X))[:n_clusters]
    centers = X[seeds]
    return centers


def init_huang(X, n_clusters, dissim, random_state):
    """Initialize centroids according to method by Huang [1997]."""
    n_attrs = X.shape[1]
    centroids = np.empty((n_clusters, n_attrs), dtype=int)
    # determine frequencies of attributes
    for iattr in range(n_attrs):
        # Sample centroids using the probabilities of attributes.
        # (I assume that's what's meant in the Huang [1998] paper; it works,
        # at least)
        # Note: sampling using population in static list with as many choices
        # as frequency counts. Since the counts are small integers,
        # memory consumption is low.
        choices = X[:, iattr]
        # So that we are consistent between Python versions,
        # each with different dict ordering.
        choices = sorted(choices)
        centroids[:, iattr] = np.random.RandomState(random_state).choice(choices, n_clusters)
    # The previously chosen centroids could result in empty clusters,
    # so set centroid to closest point in X.
    X = X.astype('int')
    for ik in range(n_clusters):
        ndx = np.argsort(dissim(X, centroids[ik]))
        # We want the centroid to be unique, if possible.
        while np.all(X[ndx[0]] == centroids, axis=1).any() and ndx.shape[0] > 1:
            ndx = np.delete(ndx, 0)
        centroids[ik] = X[ndx[0]]
    return centroids


def init_cao(X, n_clusters, dissim):
    """Initialize centroids according to method by Cao et al. [2009].
    Note: O(N * attr * n_clusters**2), so watch out with large n_clusters
    """
    n_points, n_attrs = X.shape
    centroids = np.empty((n_clusters, n_attrs), dtype='object')
    # Method is based on determining density of points.
    dens = np.zeros(n_points)
    for iattr in range(n_attrs):
        freq = defaultdict(int)
        for val in X[:, iattr]:
            freq[val] += 1
        for ipoint in range(n_points):
            dens[ipoint] += freq[X[ipoint, iattr]] / float(n_points) / float(n_attrs)

    # Choose initial centroids based on distance and density.
    centroids[0] = X[np.argmax(dens)]
    if n_clusters > 1:
        # For the remaining centroids, choose maximum dens * dissim to the
        # (already assigned) centroid with the lowest dens * dissim.
        for ik in range(1, n_clusters):
            dd = np.empty((ik, n_points))
            for ikk in range(ik):
                dd[ikk] = dissim(X, centroids[ikk]) * dens
            centroids[ik] = X[np.argmax(np.min(dd, axis=0))]
    return centroids


def InitCentroids(X, K):
    centroids = init_huang(X, K, dissim=matching_dissim, random_state=42)
    return centroids


def get_chunk_n_rows(row_bytes, *, max_n_rows=None, working_memory=None):
    if working_memory is None:
        working_memory = sklearn.get_config()["working_memory"]
    chunk_n_rows = int(working_memory * (2 ** 20) // row_bytes)
    if max_n_rows is not None:
        chunk_n_rows = min(chunk_n_rows, max_n_rows)
    if chunk_n_rows < 1:
        chunk_n_rows = 1
    return chunk_n_rows


def gen_batches(n, batch_size, *, min_batch_size=0):
    start = 0
    for _ in range(int(n // batch_size)):
        end = start + batch_size
        if end + min_batch_size > n:
            continue
        yield slice(start, end)
        start = end
    if start < n:
        yield slice(start, n)


def _argmin_min_reduce(dist, start):
    indices = dist.argmin(axis=1)
    values = dist[np.arange(dist.shape[0]), indices]  # 返回每个样本距离centroids最近的距离
    return indices, values


def pairwise_distances(X, w, centroids, gamma):
    K = len(centroids)
    distances = np.zeros((len(X), K))
    w_c = centroids * w
    v_w_v = np.sum(w_c, axis=1)
    distances = X.dot((w - 2 * w_c).T) + v_w_v.reshape(1, -1)
    return distances


def pairwise_distances_chunked(X, w, centroids, gamma, reduce_func):
    n_samples_X = X.shape[0]
    chunk_n_rows = get_chunk_n_rows(
        row_bytes=8 * len(w),
        max_n_rows=n_samples_X,
        working_memory=None,
    )
    # print("chunk_n_rows:", chunk_n_rows)
    slices = gen_batches(n_samples_X, chunk_n_rows)

    for sl in slices:
        if sl.start == 0 and sl.stop == n_samples_X:
            X_chunk = X  # enable optimised paths for X is Y
        else:
            X_chunk = X[sl]
        D_chunk = pairwise_distances(X_chunk, w, centroids, gamma)
        D_chunk = _argmin_min_reduce(D_chunk, sl.start)
        yield D_chunk


def pairwise_distances_argmin_min(X, w, centroids, gamma):
    indices, values = zip(
        *pairwise_distances_chunked(X, w, centroids, gamma, reduce_func=_argmin_min_reduce)
    )
    indices = np.concatenate(indices)
    values = np.concatenate(values)
    return indices, values


def _centers_dense(X, labels, n_clusters):
    n_features = X.shape[1]
    centers = np.zeros((n_clusters, n_features), dtype=np.float64)

    for j in range(n_features):
        centers[:, j] = np.bincount(labels, weights=X[:, j], minlength=n_clusters)
    return centers


def findClosestCentroids(X, w, centroids, gamma):
    XD = X.to_delayed().flatten().tolist()  # a list of numpy array
    func = delayed(pairwise_distances_argmin_min, pure=True, nout=2)
    blocks = [func(x, w, centroids, gamma) for x in XD]

    argmins, mins = zip(*blocks)

    argmins = [
        da.from_delayed(block, (chunksize,), np.int64)
        for block, chunksize in zip(argmins, X.chunks[0])  # X.chunks[0]是关于各个chunksize的array
    ]
    # Scikit-learn seems to always use float64
    mins = [
        da.from_delayed(block, (chunksize,), "f8")
        for block, chunksize in zip(mins, X.chunks[0])
    ]
    labels = da.concatenate(argmins)
    distances = da.concatenate(mins)
    return labels, distances


def computeCentroidsAndWeights(X, centroids, labels, K, lambda_):
    P = X.shape[1]
    r = da.blockwise(
        _centers_dense,
        "ij",
        X,
        "ij",
        labels,
        "i",
        K,
        None,
        "i",
        adjust_chunks={"i": K, "j": P},
        dtype=X.dtype,
    )
    new_centers = da.from_delayed(
        sum(r.to_delayed().flatten()), (K, P), X.dtype
    )
    counts = np.bincount(labels, minlength=K)
    # Require at least one per bucket, to avoid division by 0.
    counts = da.maximum(counts, 1)
    # counts = counts.compute()
    new_centers = new_centers.compute()
    D = np.abs(new_centers - centroids * counts[:, None])
    new_centers = np.around(new_centers / counts[:, None])

    D = D.astype(float)
    new_w = np.exp(-D / lambda_) / np.sum(np.exp(-D / lambda_), axis=1, keepdims=True)

    return new_centers, new_w


def isConvergence(costF, max_iter):
    if math.isnan(np.sum(costF)):
        return False
    index = np.size(costF)
    for i in range(index - 1):
        if costF[i] < costF[i + 1]:
            return False
    if index >= max_iter:
        print("=============reach max iter==========")
        return True
    elif costF[index - 1] == costF[index - 2] == costF[index - 3]:
        print("=============cost converge===========")
        return True
    return 'continue'


def wkmodes(X, K, max_iter, lambda_, gamma):
    n, m = X.shape
    costF = []
    w = np.ones((K, m)) / m

    # time1 = time.time()
    centroids = InitCentroids(X, K)
    # time2 = time.time()
    # print(time2 - time1)
    # print("========init centroids over=======")
    # print(centroids)

    X = da.from_array(X, chunks=(max(1, len(X) // cpu_count()), X.shape[-1]))
    # print(X.chunks[0])
    for i in range(max_iter):
        # print("iter ", i)
        labels, distances = findClosestCentroids(X, w, centroids, gamma)
        labels = labels.compute()

        time4 = time.time()
        # labels = labels.astype(np.int32)
        # print(time4 - time3)
        # print("=====compute  weights===")
        new_centriods, new_w = computeCentroidsAndWeights(X, centroids, labels, K, lambda_)
        # time5 = time.time()
        # print(time5 - time4)
        # print("=====update cost=====")

        # c = np.sum(distances.compute()) + lambda_ * np.sum(new_w * np.log(new_w))

        gap = np.ravel(centroids - new_centriods, order="K")
        shift = np.dot(gap, gap)
        if shift < 1e-7:
            # print("=====ending====")
            # print(i)
            break
        centroids = new_centriods
        w = new_w

    if shift > 1e-5:
        print("not converged")
        return False, None, None, costF, w
    return True, labels, centroids, costF, w


def get_unique(input_list):
    temp = []
    for item in input_list:
        if item not in temp:
            temp.append(item)
    return temp


class WKModes:

    def __init__(self, n_clusters=3, max_iter=100, lambda_=7.0, gamma=1):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.lambda_ = lambda_
        self.gamma = gamma

    def fit(self, X):
        self.isConverge, self.best_labels, self.best_centers, self.cost, self.w = wkmodes(
            X=X, K=self.n_clusters, max_iter=self.max_iter, lambda_=self.lambda_, gamma=self.gamma
        )
        return self

    def fit_predict(self, X, y=None):
        if self.fit(X).isConverge:
            return self.best_labels
        else:
            print('Not convergence with current parameter or centroids,Please try again')
            return None

    def get_params(self):
        return self.isConverge, self.n_clusters, self.lambda_, self.gamma, 'WKMD'

    def get_cost(self):
        return self.cost

    def get_weight(self):
        return self.w

    def get_centers(self):
        return self.best_centers


def extract_wkmodes(data_df, label_col, label_val, pos_k=None, neg_k=None, lambda_=0.1):
    """
    extract rules based on weighted kmodes, which is a clustering method

    :param data_df: pandas DataFrame, including features and label as columns
    :param label_col: column name of label
    :param label_val: value of positive class in label_col
    :param pos_k: user-defined num of clusters in positive samples, default None
    :param neg_k: user-defined num of clusters in negative samples, default None
    :param lambda_: default 0.1, control the entropy of feature weights
                    bigger value results in more uniform distribution of weights
    :return:
    """
    pos_index = data_df[data_df[label_col] == label_val].index
    neg_index = data_df[data_df[label_col] == label_val].index

    y = data_df.pop(label_col)

    binarizer = features.FeatureBinarizer(numThresh=9, negations=False, threshStr=True)  # negations：是否引入否特征
    onehot_X = binarizer.fit_transform(data_df)

    col_family = {}
    for eachcol in onehot_X.columns.values:
        tmpkey = eachcol[0]
        if tmpkey not in col_family:
            col_family[tmpkey] = [''.join(eachcol)]
        else:
            col_family[tmpkey].append(''.join(eachcol))
    onehot_X.columns = [(''.join(col)) for col in onehot_X.columns.values]
    dim_list = list(onehot_X.columns)

    ori_category_features = set()
    for each in list(onehot_X.columns):
        if '==' in each or '!=' in each:
            ori_category_features.add(each.split("==")[0])
    # print(ori_category_features)

    feature2id = {}
    id2feature = {}
    for index, feature in enumerate(dim_list):
        feature2id[feature] = index
        id2feature[index] = feature

    feature2family = {}
    for eachkey in col_family:
        for eachcol in col_family[eachkey]:
            feature2family[eachcol] = eachkey
    # print(feature2family)

    onehot_X = onehot_X.to_numpy()
    pos_data = onehot_X[pos_index, :]
    if pos_k is None:
        pos_k = max(min(len(pos_data) // 50, 50), 10)
    pos_model = WKModes(n_clusters=pos_k, max_iter=300, lambda_=lambda_, gamma=1)
    pos_model.fit(pos_data)

    neg_data = onehot_X[neg_index, :]
    if neg_k is None:
        neg_k = max(min(len(neg_data) // 50, 50), 10)
    neg_model = WKModes(n_clusters=neg_k, max_iter=300, lambda_=lambda_, gamma=1)
    neg_model.fit(neg_data)

    pos_centers = pos_model.get_centers()
    pos_weights = pos_model.get_weight()
    neg_centers = neg_model.get_centers()
    neg_weights = neg_model.get_weight()
    N_p = pos_centers.shape[0]
    N_n = neg_centers.shape[0]
    N = len(dim_list)
    candidate_rule_list = []

    center_rule_list = []

    for ii in range(N_p):
        center = pos_centers[ii]
        weight = pos_weights[ii]

        for jj in range(N_n):
            ref_center = neg_centers[jj]
            ref_weight = neg_weights[jj]

            rule = []
            neg_gain = 0
            for pp in range(N):
                if center[pp] == ref_center[pp] == 0 and weight[pp] == ref_weight[pp] == 0:
                    continue
                if center[pp] == ref_center[pp]:
                    gain = weight[pp] - ref_weight[pp]
                else:
                    gain = weight[pp] + ref_weight[pp]
                    neg_gain += ref_weight[pp]
                rule.append((dim_list[pp], center[pp], gain))
            rule = sorted(rule, key=lambda x: x[-1])

            short_rule = []
            short_item_v0 = {}
            short_item_v1 = {}
            cate_v1 = []
            cate_items = []
            cum_gain = 0
            cnt = 0
            for item in rule:
                cnt += 1
                if cum_gain + item[-1] < 1.0 * neg_gain:
                    cum_gain += item[-1]
                elif N - cnt < 31:  # N - cnt < 31
                    short_rule.append((item[0] + ":" + str(item[1])))
                    if feature2family[item[0]] in ori_category_features:
                        cate_items.append((item[0] + ":" + str(item[1])))
                        if item[1] == 1:
                            cate_v1.append(feature2family[item[0]])
                    else:
                        if item[1] == 0:
                            if feature2family[item[0]] not in short_item_v0:
                                short_item_v0[feature2family[item[0]]] = feature2id[item[0]]
                            else:
                                if short_item_v0[feature2family[item[0]]] < feature2id[item[0]]:
                                    short_item_v0[feature2family[item[0]]] = feature2id[item[0]]
                        if item[1] == 1:
                            if feature2family[item[0]] not in short_item_v1:
                                short_item_v1[feature2family[item[0]]] = feature2id[item[0]]
                            else:
                                if short_item_v1[feature2family[item[0]]] > feature2id[item[0]]:
                                    short_item_v1[feature2family[item[0]]] = feature2id[item[0]]
                if len(short_rule) > 0:
                    short_rule2 = []
                    for eachkey in short_item_v0:
                        short_rule2.append(id2feature[short_item_v0[eachkey]] + ":0.0")
                    for eachkey in short_item_v1:
                        short_rule2.append(id2feature[short_item_v1[eachkey]] + ":1.0")

                    for eachitem in cate_items:
                        fea_name = eachitem.split(":")[0]
                        if feature2family[fea_name] not in cate_v1:
                            short_rule2.append(eachitem)
                        elif float(eachitem.split(":")[1]) == 1:
                            short_rule2.append(eachitem)
                short_rule2.sort()
                short_rule.sort()
                candidate_rule_list.append(short_rule2)

    candidate_rule_list = candidate_rule_list + center_rule_list
    rule_pool = get_unique(candidate_rule_list)
    if len(rule_pool) == 0:
        print("======notice: there is no candidate rule======")
    else:
        print("num of rule pool: ", len(rule_pool))
    return rule_pool
