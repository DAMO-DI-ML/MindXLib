import copy
import pandas as pd
import logging
from multiprocessing import Pool
from pyroaring import BitMap
import numpy as np
import time
import ast
from mip import Model, xsum, minimize, BINARY, CONTINUOUS, Constr, Column, MINIMIZE
from sklearn.metrics import balanced_accuracy_score, accuracy_score

# import features
# from datautil import DatasetLoader

logging.basicConfig(
    # filename='logs/cgfpdiv.log',
    # filemode='w',
    format='[%(asctime)s %(levelname)s %(funcName)s %(lineno)s] %(message)s',
    level=logging.DEBUG,
    datefmt='%Y-%m-%d %H:%M:%S')
# logging.getLogger().setLevel(logging.DEBUG)
# logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger()


class Itemset(object):
    '''
    We use frozenset to represent an itemset.
    '''
    items: list = None
    dbs: dict = None
    labels: list = None

    @staticmethod
    def set_items(items: list):
        Itemset.items = items

    @staticmethod
    def clear_db():
        Itemset.dbs = dict()
        Itemset.labels = list()

    @staticmethod
    def set_db(l: int, db: list):
        Itemset.labels.append(l)
        Itemset.labels = sorted(Itemset.labels)
        Itemset.dbs[l] = db

    @staticmethod
    def db2idx(sign):
        base = sum([len(Itemset.dbs[l]) for l in Itemset.labels if sign>l])
        return [j+base for j, t in enumerate(Itemset.dbs[sign])]

    def __init__(self, s: set, supp=False):
        if len(s) > len(Itemset.items):
            raise Exception('num of items < {}'.format(len(s)))
        self.s = frozenset(s)
        if supp is True:
            self._cov = dict()
            for l in Itemset.labels:
                self._cov[l] = set([j for j,t in zip(Itemset.db2idx(l), Itemset.dbs[l]) if self.cover(t)])

    def __len__(self) -> int:
        return len(self.s)

    def support(self, signs: list) -> int:
        signs = set(signs)
        return sum([self._support(l) for l in Itemset.labels if l in signs])

    def _support(self, sign: int) -> int:
        return len(self._cov[sign])

    def coverage(self, signs: list) -> set:
        '''Return a set of trans ids'''
        signs = set(signs)
        return set.union(*[self._cov[l] for l in Itemset.labels if l in signs])

    def cover(self, T) -> bool:
        if len(self) > len(T):
            return False

        diff = self.s.difference(T.s)
        return True if len(diff)==0 else False

    def itemdiff(self, other):
        return self.s.difference(other.s)


class Rule(Itemset):
    '''
    Rules that contain the same set of items are considered the same.
    '''
    @staticmethod
    def quality(S: list, metric: str='kl') -> float:
        '''A modular quality'''
        if metric == 'kl':
            return sum([s.kl for s in S])
        if metric == 'acc':
            return sum([s.acc for s in S])
        raise Exception('')

    def __init__(self, s: set, l: int):
        super().__init__(s, supp=True)
        self.label = l

    def __eq__(self, other) -> bool:
        '''
        frozenset is hashable.
        '''
        return self.s == other.s and self.label == other.label

    def __hash__(self):
        return hash(self.s) ^ hash(self.label)

    @property
    def kl(self) -> int:
        '''KL distance'''
        supps = np.array([self.support([l_]) for l_ in Itemset.labels])
        if sum(supps) == 0:
            return 0

        ns = np.array([len(Itemset.dbs[l_]) for l_ in Itemset.labels])
        p = supps/sum(supps)
        p = np.where(p > 1e-9, p, 1e-9)
        q = ns/sum(ns) # q wouldn't be zero
        kl = np.sum(np.where(p != 0, p * np.log(p / q), 0))

        supp_l = self.support([self.label])
        imb_l = self.support([self.label])/sum(supps) - len(Itemset.dbs[self.label])/sum(ns)
        supp_l = supp_l if imb_l > 0 else 0
        return np.sqrt(supp_l) * kl

    @property
    def acc(self) -> float:
        '''TP / (TP + FP)'''
        dnm = self.support(Itemset.labels)
        if dnm == 0:
            return 0.0
        return self.support([self.label]) / dnm

    def overlap(self, S: list, card=False) -> float:
        if card:
            c = self.coverage(Itemset.labels)
            return [set.intersection(c, s.coverage(Itemset.labels)) for s in S]
        else:
            return sum([self._overlap(s) for s in S])

    def _overlap(self, s) -> float:
        '''Jaccard distance'''
        c = self.coverage(Itemset.labels)
        cs = s.coverage(Itemset.labels)
        cap = set.intersection(c, cs)
        if len(cap) == 0:
            return 1
        cup = set.union(c, cs)
        return 1 - len(cap) / len(cup)

    def trans(self, labels=None):
        if labels is None:
            labels = Itemset.labels
        ll = [self._cov2db(l) for l in labels]
        return [em for sl in ll for em in sl]

    def _cov2db(self, label):
        cov = self.coverage([label])
        return [t for j, t in zip(Itemset.db2idx(label), Itemset.dbs[label]) if j in cov]


class Transaction(Itemset):

    def __init__(self, s: set):
        super().__init__(s, supp=False)


def prep_db(X: pd.DataFrame, y: np.ndarray):
    '''X: Each row is a transaction'''
    Itemset.set_items(range(X.shape[1]))
    Itemset.set_items(range(X.shape[1]))
    Itemset.dbs = dict()

    for l in y.unique().tolist():
        X_ = X[y == l]
        db = _prep_db(X_ if type(X_) == np.ndarray else X_.values)
        Itemset.set_db(l, db)


def _prep_db(X: np.ndarray):
    return [Transaction(feat2item(t)) for t in X]


def feat2item(x: list):
    '''Return active items'''
    return np.nonzero(x)[0]


# split the data, row data with label_col == label_val as outlier
# the rest are inliers
def split_data(df, label_col, label_val):
    """
    Parameters
        df: pandas dataframe
        label_col: column name to represent label;
        label_val: label_col == label_val -> outlier group
    Return
        Outlier pandas dataframe, Inlier pandas dataframe
    """

    return (
        df[df[label_col] == label_val].copy().reset_index(drop=True),
        df[df[label_col] != label_val].copy().reset_index(drop=True),
    )


# format dataframe to list of itemsets (col:val)
# col are from dim_list
def get_col_val_itemset(df, dim_list):
    """
    Parameters:
        df: pandas dataframe
        dim_list: list, cols used to drill up
    Return:
        List of itemset list, all in col:val format
    """
    # col_val = pd.DataFrame(
    #     {col: str(col) + ":" for col in df[dim_list]}, index=df[dim_list].index
    # ) + df[dim_list].astype(str)

    # initial_list = col_val.values.tolist()
    # filter_list = [[j for j in i if j[-2:] != ':0'] for i in initial_list]

    # return filter_list
    return [[dim_list[i] for i in np.nonzero(t)[0]] for t in df[dim_list].values]


# create one-hot format of original data, save the bitMap rep in dict
def bitify_data(df, dim_list, pos_end):
    """
    Parameters
        df: data to be bitify
        dim_list: list, cols used to drill up
    Return
        dict, each key:value is a col and its bit result in dim_list
    """
    full_bit_dict = {'pos_bit': {}, 'neg_bit': {}}
    if type(df) == pd.core.frame.DataFrame:
        data_encode = pd.get_dummies(df[dim_list], prefix_sep=':')
        for i in data_encode.columns:
            if i[-2:] != ':0':
                whole_index = np.array(data_encode[i]).nonzero()[0]
                full_bit_dict['pos_bit'][i] = BitMap(whole_index[whole_index <= pos_end])
                full_bit_dict['neg_bit'][i] = BitMap(whole_index[whole_index > pos_end])
    else:
        index_dict = {}
        for k, each_tran in enumerate(df):
            for each_item in each_tran:
                if each_item in index_dict:
                    index_dict[each_item].append(k)
                else:
                    index_dict[each_item] = [k]
        for key in index_dict:
            full_bit_dict[key] = BitMap(index_dict[key])
    return full_bit_dict


# use data bit format to filter record index by filtering bitMap
def filter_pat_via_bit(pattern, data_bit_dict):
    """
    Parameters
        pattern: a pattern from mined result
        data_bit_dict: bited data
    Return
        BitMap result, after filter data by pattern bit
    """
    # item_list = list(pattern)
    # get count by use bit operation on full data
    # filter_bit_list = []
    # for i in range(len(item_list)):
    # for i in pattern:
    # print(item_list[i])
    # filter_bit_list.append(data_bit_dict[i])
    # print(data_bit_dict[item_list[i]])
    filter_bit_list = [data_bit_dict[i] for i in pattern]
    res = BitMap.intersection(*filter_bit_list)
    # print(res)
    # print('***************************')
    return res


# since data is compressed, use 'count' key:item to restore count value
def get_pat_cnt(pattern_bit, data_bit_dict):
    return data_bit_dict['count'][pattern_bit.to_array()].sum()


def get_pat_contr(pattern_bit, data_bit_dict):
    return data_bit_dict['contr'][pattern_bit.to_array()].sum()


# bit implement to compute for interestingness score
# use tuple as input to include all necessary parameters for multi-processing
def mp_f_bit(input_tuple):
    """
    Parameters
        input_tuple: (pattern, pattern_cnt, out_cnt, tot_cnt,
                        full_bit_dict, score_type)
        pattern: an itemset, col:val type
        pattern_cnt: weighted count of pattern from mining
        out_cnt: total weighted outliers count
        tot_cnt: total weighted count
        full_bit_dict: dict for full data, with col as key, value is bitMap
        score_type: 'risk' and 'diffScore'
    Return
        [pattern, pattern_cnt, interestingness_score]
    """
    (pattern, pattern_cnt, out_cnt, tot_cnt,
     full_bit_dict, score_type, pattern_neg_bit) = input_tuple

    # get count by use bit operation on full data
    pattern_pos_bit = filter_pat_via_bit(pattern, full_bit_dict['pos_bit'])
    if not pattern_neg_bit:
        pattern_neg_bit = filter_pat_via_bit(pattern, full_bit_dict['neg_bit'])
    # pos label index matched by pat
    # label_bit = full_bit_dict['pos_bit']
    # pat_label_bit = label_bit.intersection(pattern_bit)
    # filter_df_len = get_pat_cnt(neg_bit, full_bit_dict)
    in0_len = get_pat_cnt(pattern_neg_bit, full_bit_dict)
    # out1_len: not-pattern cnt in outlier
    # in0_len: pattern cnt in inlier
    # in1_len: not pattern cnt in inlier
    out1_len = out_cnt - pattern_cnt
    # in0_len = filter_df_len - pattern_cnt
    filter_df_len = in0_len + pattern_cnt
    in1_len = tot_cnt - out_cnt - in0_len
    # logging.info('details ({},{},{},{})'.format(pattern_cnt, in0_len, out1_len, in1_len))
    # support two score types
    if score_type == 'risk':
        comp1 = pattern_cnt / (pattern_cnt + in0_len)
        comp2 = (out1_len + 1e-4) / (out1_len + in1_len + 1e-4)
        score = comp1 / comp2
    elif score_type == 'diffScore':
        recall = pattern_cnt / out_cnt
        # logging.info('pattern_cnt {} out_cnt {}'.format(pattern_cnt, out_cnt))
        precision = pattern_cnt / filter_df_len if filter_df_len else 0
        score = 2 * precision * recall / (precision + recall)
        # logging.info('precision {} recall {}'.format(precision, recall))
    elif score_type == 'rep':
        score = (pattern_cnt - in0_len) / filter_df_len
    elif score_type == 'rep*':
        score1 = (pattern_cnt - in0_len) / filter_df_len
        comp1 = pattern_cnt / (pattern_cnt + in0_len)
        comp2 = (out1_len + 1e-4) / (out1_len + in1_len + 1e-4)
        score2 = comp1 / comp2
        score = score1 * score2

    details = (round(pattern_cnt, 2), round(in0_len, 2),
               round(out1_len, 2), round(in1_len, 2))

    return {
        'pattern': pattern,
        'details': details,
        'score': round(score, 4),
        'pos_bit': pattern_pos_bit,
        'neg_bit': pattern_neg_bit,
        'len': len(pattern)
    }
    # [pattern, pattern_cnt, details, round(score, 4)]


# multi-process to evalute score each pattern candidate
def score_candi(itemset_candi, out_cnt, tot_cnt, full_bit_dict, score_type,
                score_gap):
    """
    Parameters
        itemset_candi: list of pattern info, format[pattern, pattern_cnt]
        pattern: an itemset, col:val type
        pattern_cnt: weighted count of pattern from mining
        out_cnt: total weighted outliers count
        tot_cnt: total weighted count
        full_bit_dict: dict for full data, with col as key, value is bitMap
        score_type: 'risk' and 'diffScore'
    Return
        [pattern, pattern_cnt, interestingness_score]
    """
    # construct list of input tuples for mp
    input_tuple_list = [(i['pattern'], i['details'][0], out_cnt, tot_cnt, full_bit_dict,
                         score_type) for i in itemset_candi]
    with Pool(16) as p:
        new_res_bit = p.map(mp_f_bit, input_tuple_list)

    sort_res = sorted(new_res_bit, key=lambda x: x['score'], reverse=True)
    if sort_res:
        max_score = sort_res[0]['score']
        return [i for i in sort_res if i['score'] / max_score >= score_gap and i['details'][0] > i['details'][1]]
    else:
        return sort_res


# use bitmap to compute jaccard similarity
def jcd_dist(pattern1, pattern2, full_bit_dict):
    """
    Parameters
        pattern1, pattern2: patterns to compare
        full_bit_dict: dict for full data, with col as key, value is bitMap
    Return
        float, len of intersect data/union by two patterns
    """
    filter_bit1 = filter_pat_via_bit(pattern1, full_bit_dict['tot_bit'])
    filter_bit2 = filter_pat_via_bit(pattern2, full_bit_dict['tot_bit'])
    intersect = filter_bit1.intersection(filter_bit2)
    intersect_len = get_pat_cnt(intersect, full_bit_dict)

    union = filter_bit1.union(filter_bit2)
    union_len = get_pat_cnt(union, full_bit_dict)
    return intersect_len / union_len


# postprocess based on succinctness
def post_process(cl_res, out_num, jcd_limit, min_pat_len, full_bit_dict):
    """
    Parameters
        cl_res: list of lists, patterns including pattern, cnt, and score
        out_num: int, number of top patterns need to return
        jcd_limit: float, jaccard similirity, add new pattern when < limit
        min_pat_len: minimum len of pattern to include, to prune bad case
        full_bit_dict: dict for full data, with col as key, value is bitMap
    Return
        list of lists, final top subset patterns
    """
    cl_slim_res = []
    for q, new_pattern in enumerate(cl_res):
        if len(new_pattern[0]) < min_pat_len:
            continue
        # base case add
        if not cl_slim_res:
            cl_slim_res.append(new_pattern)
            continue
        count = 0
        # return when enough found
        if len(cl_slim_res) >= out_num:
            break
        # for each existing pattern so far
        # test if need to add new pattern
        for add_pattern in cl_slim_res:
            jcd_score = jcd_dist(new_pattern[0], add_pattern[0], full_bit_dict)
            set1 = frozenset(new_pattern[0])
            set2 = frozenset(add_pattern[0])
            if (
                    frozenset.issubset(set1, set2)
                    or frozenset.issubset(set2, set1)
                    or jcd_score > jcd_limit
            ):
                break
            else:
                count += 1
            # test pass when compare with every pattern in result so far, add
        if count == len(cl_slim_res):
            cl_slim_res.append(new_pattern)

    return cl_slim_res


# format for final output
def final_res_assemble(cl_slim_res, out_num, label_col, label_val,
                       full_bit_dict):
    """
    Parameters
        cl_slim_res: ranked top results
        out_num: number of ranked top results
        label_col: column name to represent label;
        label_val: label_col == label_val -> outlier group
        full_bit_dict: dict for full data, with col as key, value is bitMap
    Return
        pandas dataframe, with [rules, count, label_col + "_ratio"' as columns
    """
    per_res = []
    tot_match_index = None
    for q, pattern in enumerate(cl_slim_res[:out_num]):
        match_data_index = filter_pat_via_bit(pattern[0], full_bit_dict)
        if 'contr' in full_bit_dict:
            filter_df_len = get_pat_contr(match_data_index, full_bit_dict)
        else:
            filter_df_len = get_pat_cnt(match_data_index, full_bit_dict)
        # print(pattern, filter_df_len, match_data_index)
        if not tot_match_index:
            tot_match_index = match_data_index
        else:
            tot_match_index = tot_match_index.union(match_data_index)
        # get pattern intersect with label_col index bit
        label_bit = full_bit_dict['pos_bit']
        pat_label_bit = label_bit.intersection(match_data_index)
        if 'contr' in full_bit_dict:
            label_cnt = get_pat_contr(pat_label_bit, full_bit_dict)
            ratio_val = label_cnt
        else:
            label_cnt = get_pat_cnt(pat_label_bit, full_bit_dict)
            ratio_val = label_cnt / filter_df_len

        per_res.append(
            [pattern[0]]
            + [filter_df_len, ratio_val, pattern[-2], pattern[-1]]
        )
    if not per_res:
        return cl_slim_res
    out_df = pd.DataFrame(per_res)
    out_df.columns = ["rules", "count", label_col + "_ratio", "details", "score"]
    out_df["rules"] = out_df["rules"].astype(str)
    tot_match_index = tot_match_index.intersection(label_bit)
    tot_matched_label = get_pat_cnt(tot_match_index, full_bit_dict)
    # logging.info('tot matched label {}'.format(tot_matched_label))
    return out_df


# sort and build col_to_index + index_to_col dict
def build_ref_dict(full_bit_dict):
    """
    Parameters
        full_bit_dict: dict for full data, with col as key, value is bitMap
    Return
        dict, dict, column to index map, and vice verse
    """
    sorted_keys = sorted(full_bit_dict['pos_bit'])
    col_to_index, index_to_col = {}, {}
    for k, i in enumerate(sorted_keys):
        col_to_index[i] = k
        index_to_col[k] = i

    return col_to_index, index_to_col


# pattern bitify, extract from bit dict
def pat_bit(p, col_to_index):
    """
    Parameters
        p: pattern
        col_to_index: dict map item (col name) to its index
    Return
        BitMap, rep for pattern
    """
    p_list = []
    for i in p:
        p_list.append(col_to_index[i])
    return BitMap(p_list)


def pattern_to_dict(p):
    filter_dict = {}
    for i in sorted(list(p)):
        filter_dict[i.split(':')[0]] = i.split(':')[1]
    return filter_dict


# filter_dict1 = pattern_to_dict(pat)
# filter_df = res.loc[(res[list(filter_dict1)] == pd.Series(filter_dict1)).all(axis=1)]

class DIVER():
    def __init__(self, label_col, label_val, pos_beta=1.5, overlap_beta_=0.2,
                 complexity_cost=0.00001,dim_list=None,sup_ratio=0.01,
                write_model=False, disable_log=True, cache_ind=False):
        self.label_col =label_col
        self.label_val = label_val
        self.pos_beta = pos_beta
        self.overlap_beta_ = overlap_beta_
        self.complexity_cost = complexity_cost
        self.dim_list = dim_list
        self.sup_ratio = sup_ratio
        self.write_model = write_model
        self.disable_log = disable_log
        self.cache_ind = cache_ind

    def fit(self, X, y):
        data_df = pd.concat((X, y.to_frame()), axis=1)
        label_cnt = data_df[self.label_col].value_counts()
        label_info = [(label_cnt.index[k], label_val) for (k, label_val) in enumerate(label_cnt)]
        sort_label_info = sorted(label_info, key=lambda x: x[1])
        # print(sort_label_info)  #  [(0, 332), (1, 626)]
        # print(sort_label_info[-1][0])
        self.default_label = 0

        output_rule = []
        diver_rule = []
        if self.dim_list is None:
            self.dim_list = list(data_df.columns)
            self.dim_list.remove(self.label_col)
        Itemset.clear_db()
        prep_db(data_df[self.dim_list],data_df[self.label_col])
        res = self.drillUp(data_df, self.label_col, self.label_val, self.dim_list, self.sup_ratio,self.pos_beta,
                      self.overlap_beta_,self.complexity_cost,self.write_model,self.disable_log,self.cache_ind)
        for each_rule in res['lp_res']['short_rule']:
            print(each_rule, self.label_val)
            # print([self.dim_list.index(i) for i in each_rule])
            output_rule.append(each_rule)
            add = Rule([self.dim_list.index(i) for i in each_rule], self.label_val)
            diver_rule.append(add)
        self.return_rule = output_rule
        self.diver_rule = diver_rule
        return self

    def predict(self, X_test):
        Xtest_list = [Transaction(feat2item(t)) for t in X_test.values]
        ypredict_diver = predict_all(self.diver_rule, self.default_label, Xtest_list)
        return ypredict_diver
        # bacc = balanced_accuracy_score(y_test, ypredict_diver)
        # acc = accuracy_score(y_test, ypredict_diver)


    # core drill up procedure
    def drillUp(self, c_df, label_col, label_val, dim_list,
                sup_ratio,
                pos_beta, overlap_beta_, complexity_cost,
                write_model, disable_log, cache_ind):
        """
        Parameters
            c_df: dataframe after preprocessed, groupby, add 'count' col
            label_col: column name to represent label;
            label_val: label_col == label_val -> outlier group
            dim_list: list, cols used to drill up
            sup_ratio: support ratio
            min_pat_len: minimum len of pattern to include, to prune bad case
            score_type: 'risk' and 'diffScore'
        Return
            a pandas dataframe
        """
        c_df['count'] = 1
        c_df['col_has_other_map_no_other'] = ''

        global overlap_beta
        overlap_beta = overlap_beta_

        logger.disabled = disable_log

        c_df['count'] = c_df['count'].astype(float)

        # throw away bad cols
        bad_cols = []
        num = c_df.shape[0]
        for col in dim_list:
            num_lvls = c_df[col].nunique(dropna=True)
            if num_lvls <= 1:
                bad_cols.append(col)
        dim_list = [i for i in dim_list if i not in bad_cols]

        # order data, first come pos, then neg
        c_df.loc[c_df[label_col] == label_val, 'label'] = '1'
        label_val = '1'
        c_df.loc[c_df[label_col] != label_val, 'label'] = '0'
        c_df = c_df.sort_values(by=['label'], ascending=False).reset_index()
        tot_index_list = list(c_df['index'].values)
        c_df = c_df[[i for i in c_df.columns if i != 'index']]

        # step 1. split the data
        outlier_df, inlier_df = split_data(c_df, label_col, label_val)
        # print(outlier_df.head(5))
        logging.info("Data split finished!")
        logging.info("Ouliter shape {} and cnt {}".format(outlier_df.shape[0],
                                                          outlier_df['count'].sum()))

        # step 2. get bit format for dataframe
        outlier_bit_dict = bitify_data(outlier_df, dim_list, outlier_df.shape[0] - 1)
        # print(outlier_bit_dict)
        outlier_bit_dict_key = list(outlier_bit_dict['pos_bit'].keys()).copy()
        outlier_bit_dict['count'] = np.array(outlier_df['count'])
        # initial item-count dict for algo
        init_dict = {}
        for key in outlier_bit_dict_key:
            # print(key)
            bit_to_array = outlier_bit_dict['pos_bit'][key].to_array()
            init_dict[key] = outlier_bit_dict['count'][bit_to_array].sum()

        pos_index = BitMap(c_df[c_df[label_col] == label_val].index)
        neg_index = BitMap(c_df[c_df[label_col] != label_val].index)

        global full_bit_dict
        full_bit_dict = bitify_data(c_df, dim_list, pos_index[-1])
        full_bit_dict['count'] = np.array(c_df['count'])
        if 'contr' in c_df.columns:
            full_bit_dict['contr'] = np.array(c_df['contr'])
        full_bit_dict['tot_bit'] = BitMap(list(c_df.index))
        full_bit_dict['tot_pos_bit'] = pos_index
        full_bit_dict['tot_neg_bit'] = neg_index
        # get to know the covered data bit info in this run
        cover_index = set()
        global cover_index_tobit
        cover_index_tobit = BitMap([tot_index_list.index(i) for i in cover_index])

        logging.info("Data to bitmap finished! tot keys {}".format(
            len(full_bit_dict['pos_bit'])))
        global col_to_index, index_to_col
        col_to_index, index_to_col = build_ref_dict(full_bit_dict)

        # step 3. fp-growth for both oulier and inlier
        outlier_list = get_col_val_itemset(outlier_df, dim_list)
        outlier_weight = list(outlier_df['count'].values)
        global minsup
        minsup = outlier_df['count'].sum() * sup_ratio

        global out_cnt, tot_cnt
        out_cnt = outlier_df['count'].sum()
        tot_cnt = c_df['count'].sum()

        global fc_list, fc_cnt, max_score, time_profile
        fc_list, fc_cnt, max_score = [], [0], [-float("inf")]
        global lp_list, lp_res_cache, redu_cost_candi
        lp_list, lp_res_cache = {'lp_list': []}, {}
        redu_cost_candi = []

        # global cache tree
        global cache_tree, col_var, col_list, cache
        cache_tree, col_var, col_list, cache = {}, [], [], cache_ind
        # global best redu cost and dual var
        global min_redu_cost, rule_cost_coef
        # global best var and express var list
        global best_var_list, express_var_list
        global mu_array, mu_array_pos, mu_array_neg_index, mu_array_pos_index

        cur_model = None
        new_var = []
        # construct single item var
        for k_i, i in enumerate(full_bit_dict['pos_bit']):
            add_pos_bit = full_bit_dict['pos_bit'][i]
            add_neg_bit = full_bit_dict['neg_bit'][i]
            if cover_index_tobit:
                add_all_bit = add_pos_bit.union(add_neg_bit)
                overlap_cover_coef = len(add_all_bit.intersection(cover_index_tobit))
            else:
                overlap_cover_coef = 0
            add_var = {'id': 'x' + str(k_i), 'pattern': [i], 'pos_bit': add_pos_bit, 'neg_bit': add_neg_bit,
                       'neg_rule_coef': len(add_neg_bit),
                       'pos_rule_coef': len(add_pos_bit),
                       'overlap_cover_coef': overlap_cover_coef,
                       'pat_len': len([i])}
            new_var.append(add_var)
            col_var.append(set([i]))
        # print('single item new_var {} col_var {}'.format(len(new_var), len(col_var)))
        # cgfp main loop
        n = len(full_bit_dict['tot_bit'])
        rule_cost_coef = complexity_cost * n

        start_time = time.time()
        find_better_var = True
        global iteration, neg_var_cnt
        iteration = [0]
        mu_list = []
        return_col_list = []
        time_profile_list = []
        solve_profile_list = []
        # for k in range(5):
        no_improv_cnt = 0
        while find_better_var:
            logging.debug('****************************')
            logging.debug('****************************')
            # print('tot vars {}'.format(len(col_var)))
            solve_profile = {'cnt': 0.0, 'time': 0.0, 'vars': 0}
            # solve RMLP with new vars
            print('iteration {}'.format(iteration[0]))

            if cur_model is None:
                # print('solve inital single item LP')
                prev_add_new_num = 0
                prev_obj = None
            print('add new vars {}'.format(len(new_var)))
            lp_res = solve_mip(cur_model, new_var, True, pos_beta, overlap_beta, write_model)
            col_list += copy.deepcopy(new_var)
            return_col_list.append(copy.deepcopy(col_list))
            mu_array = lp_res['mu_array']
            mu_array_pos = np.array([0 if i < 0 else i for i in mu_array])
            mu_array_neg_index = BitMap(np.where(mu_array < 0)[0])
            mu_array_pos_index = BitMap(np.where(mu_array >= 0)[0])
            mu_list.append(mu_array)
            cur_model = lp_res['m_object']
            obj = cur_model.objective_value
            solve_profile['cnt'] += 1
            solve_profile['time'] += lp_res['mip_time']
            solve_profile['vars'] = len(cur_model.vars) - len(full_bit_dict['tot_pos_bit'])
            solve_profile_list.append(solve_profile.copy())
            print('obtain obj value so far {}'.format(obj))
            if prev_obj is not None:
                obj_gap = np.abs(obj - prev_obj)
                if obj_gap < 1e-4:
                    # print('optimal no improve')
                    no_improv_cnt += 1
                else:
                    no_improv_cnt = 0
                if no_improv_cnt >= 2:
                    break
            prev_obj = obj

            rules_num = len(cur_model.vars) - 2 * len(full_bit_dict['tot_pos_bit'])
            # print('this many var {} in model'.format(rules_num))
            if rules_num > 3000:
                break

            # fresh the min_redu_cost for mining use
            min_redu_cost = [float('inf')]
            # fresh cond node info for new start
            cond_node_info = {'list': [], 'pos_bit': BitMap([]), 'neg_bit': BitMap([]), 'cnt': 0.0}
            # fresh global best_var_list and express_var_list
            best_var_list, express_var_list = [], []
            # fresh time_profile
            time_profile = ini_time_profile()
            neg_var_cnt = [0]

            # main mining to find best redu cost candidate
            fp_start = time.time()
            outlier_fp_tree = cl_fptree(outlier_list, outlier_weight, cond_node_info, init_dict, True)
            outlier_fp_tree.findfqt()
            # save time profile
            outlier_fp_tree.time_profile['fp_time'] = round(time.time() - fp_start, 3)
            time_profile_list.append(outlier_fp_tree.time_profile.copy())
            # check if good res exists
            best_var = outlier_fp_tree.best_var_list
            if min_redu_cost[0] / n >= -1e-5:
                # print('no good redu cost')
                best_var = []
                good_res = []
            else:
                express_var = outlier_fp_tree.express_var_list
                # express_var = []
                more_cover_var = []
                sort_express = sorted(express_var, key=lambda x: x['score'])
                cover_pos_index = best_var[0]['pos_bit'].intersection(mu_array_pos_index)
                for each_var in sort_express:
                    new_cover_part = each_var['pos_bit'].intersection(mu_array_pos_index.difference(cover_pos_index))
                    if new_cover_part:
                        more_cover_var.append(each_var)
                        cover_pos_index = cover_pos_index.union(new_cover_part)

                # good_res = best_var + express_var
                good_res = best_var + more_cover_var
                # print('best neg cost candi is {} with score {}'.format(sorted(best_var[0]['pattern']), min_redu_cost[0]))
            new_var = []
            for i in good_res:
                if set(i['pattern']) not in col_var and i not in new_var:
                    col_var.append(set(i['pattern'].copy()))
                    new_var.append(copy.deepcopy(i))

            # print('best neg cost candi score {}'.format(min_redu_cost[0]))
            # logging.info('best neg cost candi is {}'.format(best_var))
            # logging.info('good 0 neg cnt candi is {}'.format(express_var))
            # logging.info('not in model ones are {} {}'.format(len(new_var), new_var))

            prev_add_new_num = len(new_var)

            if not new_var:
                find_better_var = False
                # print('stop for no improved vars found')
            # find_better_var = False
        # print(time.time() - start_time)
        short_rule_index = lp_res['short_rule_index']
        # n_pos = len(full_bit_dict['tot_pos_bit'])
        if short_rule_index:
            union_pos_bit = BitMap.union(*[col_list[j]['pos_bit'] for j in short_rule_index])
            union_neg_bit = BitMap.union(*[col_list[j]['neg_bit'] for j in short_rule_index])
            union_bit = union_pos_bit.union(union_neg_bit)
            this_cover_index = np.array(tot_index_list)[union_bit.to_array()]
        else:
            this_cover_index = []

        return {'lp_res': lp_res, 'mu_list': mu_list, 'return_col_list': return_col_list, 'full_bit_dict': full_bit_dict,
                'time_profile_list': time_profile_list, 'solve_profile_list': solve_profile_list
                }

def predict(rules, default, x) -> bool:
    for r in rules:
        if r.cover(x):
            return r.label
    return default

def predict_all(rule, default, X: list) -> list:
    return np.array([predict(rule, default, x) for x in X])

def ini_time_profile():
    time_profile = {
        'fq_call': 0,
        'read_cache': {'time': 0.0, 'cnt': 0},
        'construct': {'time': 0.0, 'cnt': 0},
        'construct_sort': 0.0,
        'construct_buildTree': 0.0,
        'node_table': 0.0,
        'each_data_sort': 0.0,
        'each_data_bit_build': 0.0,
        'bit_cum_time': 0.0,
        'tot_condtree': 0,
        'prune_condtree': 0,
        'use_condtree': {'time': 0.0, 'cnt': 0},
        'condtree_ub': {'time': 0.0, 'cnt': 0},
        'extract_cl': {'time': 0.0, 'cnt': 0},
        'get_itm_cnt': {'time': 0.0, 'cnt': 0},
        'redu_prune': {'cnt': 0, 'volume': 0},
        'prune_fq_call': 0,
        'cond_bit_prep': 0.0,
    }
    return time_profile


# compute pattern upper bound info
def get_score_ub(pattern_cnt, pat_lb, out_cnt, tot_cnt, score_type):
    """
    Parameters
        pattern_cnt: pat upper bound in outlier
        pat_lb: pat lower bound in inlier
        out_cnt: outlier count sum
        tot_cnt: total count sum
    Return
        an upper bound risk score
    """
    # out1_len: not-pattern cnt in outlier
    # in0_len: pattern cnt in inlier
    # in1_len: not pattern cnt in inlier
    out1_len = out_cnt - pattern_cnt
    in0_len = pat_lb
    in1_len = tot_cnt - out_cnt - in0_len
    # comp1 = pattern_cnt/(pattern_cnt+in0_len)
    # # print(pattern_cnt, in0_len)
    # comp2 = (out1_len+1e-4)/(out1_len+in1_len+1e-4)
    # score = comp1/comp2
    if score_type == 'risk':
        comp1 = pattern_cnt / (pattern_cnt + in0_len)
        comp2 = (out1_len + 1e-4) / (out1_len + in1_len + 1e-4)
        score = comp1 / comp2
    elif score_type == 'diffScore':
        recall = pattern_cnt / out_cnt
        precision = pattern_cnt / (pattern_cnt + in0_len)
        score = 2 * precision * recall / (precision + recall)
    elif score_type == 'rep':
        score = (pattern_cnt - in0_len) / (pattern_cnt + in0_len)
    elif score_type == 'rep*':
        score1 = (pattern_cnt - in0_len) / (pattern_cnt + in0_len)
        comp1 = pattern_cnt / (pattern_cnt + in0_len)
        comp2 = (out1_len + 1e-4) / (out1_len + in1_len + 1e-4)
        score2 = comp1 / comp2
        score = score1 * score2

    return round(score, 4)


"""
Main tree search class and method
"""


# fp-growth for closed itemset
class cl_node:
    def __init__(self, itm, itm_count, parent, link):
        self.itm = itm
        self.itm_count = itm_count
        self.parent = parent
        self.link = link
        self.children = {}
        self.pos_bit = None
        self.neg_bit = None


class cl_fptree:
    def __init__(self, data, freq_weight, cond_node_info, itm_dict_in,
                 first_build
                 ):
        # raw data and minminual support
        # print(data)
        self.freq_weight = freq_weight
        self.fc_list = fc_list
        self.fc_cnt = fc_cnt
        self.cond_node_info = copy.deepcopy(cond_node_info)
        self.max_score = max_score
        self.first_build = first_build
        self.lp_list = lp_list
        self.time_profile = time_profile
        self.lp_res_cache = lp_res_cache
        self.redu_cost_candi = redu_cost_candi
        self.neg_var_cnt = neg_var_cnt

        # below is new for cgfp
        self.best_var_list = best_var_list
        self.express_var_list = express_var_list

        self.time_profile['fq_call'] += 1

        logging.debug('cur tree read in cond info is {}'.format(self.cond_node_info))
        # check if current cond tree is cached before use
        self.cond_tree_name = repr(self.cond_node_info['list'])
        name = self.cond_tree_name
        if name not in cache_tree:
            logging.debug('cond tree {} not found, initialize'.format(name))
            self.ini_tree(data, itm_dict_in)
        else:
            logging.debug('cond tree {} found, read its cache data'.format(name))
            read_start = time.time()
            self.root = cache_tree[name]['root']
            # run get itm dict again to not miss extract step
            self.itm_dict = cache_tree[name]['itm_dict_in']
            # self.get_itm_dict(data, itm_dict_in)
            self.node_table = cache_tree[name]['node_table']
            self.node_table_ref = cache_tree[name]['node_table_ref']
            self.extract_list_step()
            self.time_profile['read_cache']['cnt'] += 1
            self.time_profile['read_cache']['time'] += round(time.time() - read_start, 4)

        # check1 = max(mu_array[self.root.pos_bit.to_array()])
        # check2 = (rule_cost_coef-min_redu_cost[0])/(check1 - overlap_beta)
        # check3 = mu_array[self.root.pos_bit.to_array()].sum()
        # # if check2 > minsup:
        # print(name, check1, check2, minsup, (check3+min_redu_cost[0])/overlap_beta, len(self.root.pos_bit))

    # initialize a new tree component
    def ini_tree(self, data, itm_dict_in):
        # null root
        self.root = cl_node("Null", 1.0, None, None)
        if not self.cond_node_info['list']:
            self.root.neg_bit = full_bit_dict['tot_neg_bit']
            self.root.pos_bit = full_bit_dict['tot_pos_bit']
        else:
            self.root.neg_bit = self.cond_node_info['neg_bit']
            self.root.pos_bit = self.cond_node_info['pos_bit']
        # cache tree name
        if cache:
            cache_tree[self.cond_tree_name] = {}
            cache_tree[self.cond_tree_name]['root'] = self.root

        # node table containing link of all nodes of same item
        self.node_table = []
        self.node_table_ref = {}
        # dictionary contaiing item more than the minsupport count
        # with des order
        self.itm_sort_dict = []
        # dictionaly containing item and the support count
        self.itm_dict = {}
        # dictionary with item and it's postion of the support count rank
        self.itm_order_dict = {}
        # get item key:count and extract closed set
        self.get_itm_dict(data, itm_dict_in)
        # extract shortcut step, short the item_dict lengh
        self.extract_list_step()
        self.construct(data)

    # get item key:count dict ready
    def get_itm_dict(self, data, itm_dict_in):
        start_time = time.time()
        if not itm_dict_in:
            # get support count for all item
            for k, tran in enumerate(data):
                for itm in tran:
                    if itm in self.itm_dict:
                        self.itm_dict[itm] += self.freq_weight[k]
                    else:
                        self.itm_dict[itm] = self.freq_weight[k]
        else:
            self.itm_dict = itm_dict_in.copy()

        # cache this for use
        name = self.cond_tree_name
        if cache:
            cache_tree[name]['itm_dict_in'] = itm_dict_in.copy()
        itm_cnt_end = time.time()
        self.time_profile['get_itm_cnt']['time'] += itm_cnt_end - start_time
        # print(self.itm_dict)

    # extract closed set list with relaxation
    def extract_list_step(self):
        start_time = time.time()
        # extract pattern in every trans
        if not self.cond_node_info['list']:
            cond_node_cnt = sum(self.freq_weight)
        else:
            cond_node_cnt = self.cond_node_info['cnt']

        extract_list = []
        for each_item in self.itm_dict:
            this_item_cnt = self.itm_dict[each_item]
            if this_item_cnt == cond_node_cnt:
                logging.debug('found same cnt item cnt {} and cond node cnt \
                             {}'.format(this_item_cnt, cond_node_cnt))
                extract_list.append(each_item)

        # update cond node
        for item_i in extract_list:
            self.itm_dict.pop(item_i)
            self.update_cond_node_info(item_i, cond_node_cnt)

        cl_time_end = time.time()
        cl_time = cl_time_end - start_time
        self.time_profile['extract_cl']['time'] += cl_time

    # fun to update conditional tree info
    def update_cond_node_info(self, new_col, new_col_cnt):
        logging.info('updating cond node bit {}'.format(new_col))
        new_col_pos_bit = full_bit_dict['pos_bit'][new_col]
        new_col_neg_bit = full_bit_dict['neg_bit'][new_col]
        # pop cached res since that tree is not actually built
        if self.cond_tree_name in cache_tree:
            cache_tree.pop(self.cond_tree_name)
        self.cond_node_info['list'].append(new_col)
        # update name as well
        self.cond_tree_name = repr(self.cond_node_info['list'])

        # update corresponding node info
        self.update_single_info(self.cond_node_info, 'pos_bit', new_col_pos_bit)
        self.update_single_info(self.cond_node_info, 'neg_bit', new_col_neg_bit)
        self.cond_node_info['cnt'] = new_col_cnt

        # check the new cond pat is good to add
        pat = self.cond_node_info['list'].copy()
        freq = self.cond_node_info['cnt']
        pos_bit = self.cond_node_info['pos_bit']
        neg_bit = self.cond_node_info['neg_bit']
        add_ind = self.check_if_add_pat(pat, freq, pos_bit, neg_bit)

    # sort good candi and create node table
    def create_node_table(self):
        cl_time_end = time.time()
        itemlist = list(self.itm_dict)
        # prune all the world with < min support count
        for itm in itemlist:
            if self.itm_dict[itm] < minsup:
                del self.itm_dict[itm]
        # sort the remaing items des, with first item count than work#id
        self.itm_sort_dict = sorted(self.itm_dict.items(),
                                    key=lambda x: (-x[1], x[0]))
        logging.debug('item count sort dict is {}'.format(self.itm_sort_dict))
        # create a table containing item, itemcount and
        # all link node of that item
        t = 0
        for i in self.itm_sort_dict:
            item_n = i[0]
            item_c = i[1]
            self.itm_order_dict[item_n] = t
            self.node_table_ref[item_n] = t
            t += 1
            iteminfo = {"itemn": item_n, "itemcc": item_c, "linknode": None,
                        "last_link": None, "min_neg": float('inf'), "min_pos": float('inf')
                        }
            self.node_table.append(iteminfo)
        # save complete item sort when first build
        if self.first_build:
            logging.info('first build, has {} itm'.format(len(self.itm_order_dict)))
            # self.first_build = False
            # self.fc_cnt = [0]
        construct_sort_end = time.time()
        self.time_profile['construct_sort'] += construct_sort_end - cl_time_end

    # construct a tree
    def construct(self, data):

        cl_time_end = time.time()
        # pre construct
        self.create_node_table()
        construct_sort_end = time.time()

        # construct fptree line by line
        for k, line in enumerate(data):
            each_data_start = time.time()
            bit_cum_time = 0.0
            supitem = []
            for itm in line:
                # only keep items with support count higher than minsupport
                if itm in self.itm_dict:
                    supitem.append(itm)
            # insert items to the fp tree
            if len(supitem) > 0:
                # reorder the items
                sortsupitem = sorted(supitem,
                                     key=lambda k: self.itm_order_dict[k])
                # self.itm_line_sort.append(sortsupitem)
                each_data_sort_end = time.time()
                each_data_sort = each_data_sort_end - each_data_start
                # enter the item one by one from begining
                R = self.root
                for i in sortsupitem:
                    if i in R.children:
                        R.children[i].itm_count += self.freq_weight[k]
                        R = R.children[i]
                    else:
                        R.children[i] = cl_node(
                            i, self.freq_weight[k], R, None
                        )
                        bit_cum_start = time.time()
                        # R.children[i].pos_bit = full_bit_dict['pos_bit'][i].intersection(R.pos_bit)
                        R.children[i].neg_bit = full_bit_dict['neg_bit'][i].intersection(R.neg_bit)
                        R.children[i].pos_bit = full_bit_dict['pos_bit'][i].intersection(R.pos_bit)
                        bit_cum_time += time.time() - bit_cum_start
                        R = R.children[i]
                        # link this node to node_table
                        node_table_start = time.time()
                        iteminfo = self.node_table[self.node_table_ref[R.itm]]
                        # find the last node of the  node linklist
                        if iteminfo["linknode"] is None:
                            iteminfo["linknode"] = R
                        else:
                            iter_node = iteminfo["lastlink"]
                            iter_node.link = R
                        iteminfo["lastlink"] = R

                        if iteminfo["min_neg"] > 0:
                            new_neg_cnt = get_pat_cnt(R.neg_bit, full_bit_dict)
                            if new_neg_cnt < iteminfo["min_neg"]:
                                iteminfo["min_neg"] = new_neg_cnt

                        if iteminfo["min_pos"] > 0:
                            new_pos_cnt = get_pat_cnt(R.pos_bit, full_bit_dict)
                            if new_pos_cnt < iteminfo["min_pos"]:
                                iteminfo["min_pos"] = new_pos_cnt

                        self.time_profile['node_table'] += time.time() - node_table_start

                each_data_bit_build = time.time() - each_data_sort_end
                self.time_profile['each_data_sort'] += each_data_sort
                self.time_profile['each_data_bit_build'] += each_data_bit_build
                self.time_profile['bit_cum_time'] += bit_cum_time

        # self.dir_get_fc()
        self.time_profile['construct']['cnt'] += 1
        self.time_profile['construct']['time'] += round(time.time() - cl_time_end, 4)
        self.time_profile['construct_buildTree'] += round(time.time() - construct_sort_end, 4)

        if self.first_build:
            self.first_build = False

        # cache tree node table and ref
        name = self.cond_tree_name
        if cache:
            if name not in cache_tree:
                cache_tree[name] = {'root': self.root}
            cache_tree[name]['node_table'] = self.node_table
            cache_tree[name]['node_table_ref'] = self.node_table_ref

    # cond tree prune check
    def cond_prune_check(self, N):
        cond_prune_check_start = time.time()
        # get min inlier count from note table
        this_item_info = self.node_table[self.node_table_ref[N.itm]]
        min_neg = this_item_info['min_neg']
        min_pos = this_item_info['min_pos']

        # compute reducted cost bound
        if self.cond_node_info['pos_bit']:
            pos_bit_input = self.cond_node_info['pos_bit'].intersection(full_bit_dict['pos_bit'][N.itm])
        else:
            pos_bit_input = full_bit_dict['pos_bit'][N.itm]
        # get tree lower bound to check pruning
        tree_cost_lb = tree_redu_cost(min_neg, min_pos, overlap_beta, pos_bit_input,
                                      1 + len(self.cond_node_info['list']),
                                      'single', 'bound')

        self.time_profile['condtree_ub']['time'] += time.time() - cond_prune_check_start
        self.time_profile['condtree_ub']['cnt'] += 1

        logging.info('redu_cost_lb {} min_redu_cost {}'.format(tree_cost_lb, min_redu_cost))
        # case when this branch can not be pruned
        if tree_cost_lb < 0 and tree_cost_lb < min_redu_cost[0]:
            return False
        if tree_cost_lb >= 0:
            self.time_profile['redu_prune']['cnt'] += 1
            logging.debug('{} pruned because of positive cost'.format(self.cond_tree_name))
        elif tree_cost_lb >= min_redu_cost[0]:
            self.time_profile['redu_prune']['cnt'] += 1
            logging.debug('{} pruned because of worse than min cost'.format(self.cond_tree_name))
        else:
            logging.info('other reason pruned')
        return True

    # create transactions for conditinal tree
    def condtreetran(self, N):
        start_time0 = time.time()
        self.time_profile['cond_bit_prep'] += time.time() - start_time0
        # logging.info('current cond node bit {}'.format(self.cond_node_bit))
        start_time = time.time()
        if N.parent is None:
            self.time_profile['prune_condtree'] += 1
            return None

        # pruning start here
        check_res = self.cond_prune_check(N)
        # cond tree pruned
        if check_res:
            self.time_profile['prune_condtree'] += 1
            logging.info('this pass is pruned')
            return [], [], {}
        # not pruned, need to recursive going
        else:
            logging.info('not pruned, collect cond data')
            condtreeline = []
            cond_freq_weight = []
            cond_dict = {}

            # starting from the leaf node reverse add item till hit root
            while N is not None:
                line = []
                PN = N.parent
                # line_bit_list = [full_bit_dict[N.itm]]
                while PN.parent is not None:
                    line.append(PN.itm)
                    if PN.itm in cond_dict:
                        cur_cnt = cond_dict[PN.itm]
                    else:
                        cur_cnt = 0
                    cond_dict[PN.itm] = cur_cnt + N.itm_count
                    PN = PN.parent
                line = line[::-1]
                condtreeline.append(line)
                cond_freq_weight.append(N.itm_count)

                N = N.link

            self.time_profile['use_condtree']['cnt'] += 1
            self.time_profile['use_condtree']['time'] += time.time() - start_time
            return condtreeline, cond_freq_weight, cond_dict

    # Find frequent item list by creating conditional tree
    def findfqt(self, parentnode=None):
        if len(list(self.root.children.keys())) == 0:
            return None

        logging.debug('<< cur self cond node is {}'.format(self.cond_node_info['list']))
        logging.debug('<< fed parentnode is {}'.format(parentnode))

        # whole tree prune bound try
        if self.cond_node_info['list']:
            pos_bit_input = self.cond_node_info['pos_bit']
        else:
            pos_bit_input = full_bit_dict['tot_pos_bit']
        # get loose tree lb
        tree_cost_lb = tree_redu_cost(0, minsup, overlap_beta, pos_bit_input, len(self.cond_node_info['list']),
                                      'full', 'bound')

        # case when lb is validated
        if tree_cost_lb >= min_redu_cost[0]:
            logging.debug('cur tree not worth going, pruned whole')
            self.time_profile['prune_fq_call'] += 1
            return

        logging.debug('cur tree worth going, continue')
        # whole tree can not be pruned, loop one by one
        revtable = self.node_table[::-1]
        # logging.debug('node table order {}'.format(revtable))
        for n in revtable:
            self.time_profile['tot_condtree'] += 1
            fqset = [set(), 0]
            if parentnode is None:
                fqset[0] = {
                    n["itemn"],
                }
            else:
                fqset[0] = {n["itemn"]}.union(parentnode[0])
            fqset[1] = n["itemcc"]
            logging.debug('cur prefix parent {}'.format(parentnode))
            logging.debug('extending node {}'.format(n['itemn']))
            logging.debug('cur fqset fix {}'.format(fqset))

            condtran, condtran_freq_weight, cond_dict = self.condtreetran(n["linknode"])

            logging.debug('new cond data len {}'.format(len(condtran)))
            # keep going confirmed
            if condtran:
                # recursively build the conditinal fp tree
                add_n = n.copy()
                # add_n.pop("linknode")
                logging.debug('connect new node {} to recursive'.format(add_n['itemn']))
                new_cond_node_info = copy.deepcopy(self.cond_node_info)
                new_cond_node_info['list'].append(add_n['itemn'])
                pos_bit = full_bit_dict['pos_bit'][add_n['itemn']]
                neg_bit = full_bit_dict['neg_bit'][add_n['itemn']]

                # update corresponding component in node info
                self.update_single_info(new_cond_node_info, 'pos_bit', pos_bit)
                self.update_single_info(new_cond_node_info, 'neg_bit', neg_bit)
                new_cond_node_info['cnt'] = add_n['itemcc']

                # check the new cond pat is good to add
                pat = new_cond_node_info['list'].copy()
                freq = new_cond_node_info['cnt']
                pos_bit = new_cond_node_info['pos_bit']
                neg_bit = new_cond_node_info['neg_bit']
                add_ind = self.check_if_add_pat(pat, freq, pos_bit, neg_bit)

                # only continue is worth going
                # if add_ind:
                logging.debug('continue build cond tree')
                contree = cl_fptree(
                    condtran,
                    condtran_freq_weight,
                    new_cond_node_info,
                    cond_dict.copy(),
                    self.first_build
                )
                contree.findfqt(fqset)

        return

    # fun to check if add a proposed candi
    def check_if_add_pat(self, pat, freq, pos_bit, neg_bit):
        # remove check existing sol as their redu is different
        logging.debug('check if add pat {}'.format(set(pat)))
        if set(pat) in col_var:
            logging.debug('existing in model')
            # return False

        # update min cost part
        neg_cnt = get_pat_cnt(neg_bit, full_bit_dict)
        new_score = tree_redu_cost(neg_cnt, freq, overlap_beta, pos_bit, len(pat), 'single', 'exact')
        if cover_index_tobit:
            add_all_bit = pos_bit.union(neg_bit)
            overlap_cover_coef = len(add_all_bit.intersection(cover_index_tobit))
            new_score += overlap_beta * overlap_cover_coef
        logging.debug('cur state new_score {} len {}'.format(new_score, len(pat)))

        add_pat = self.generate_candi_format(pat, freq, pos_bit, neg_bit, new_score)

        # still add good candi if it is perfect classify
        # if neg_cnt == 0: # and self.neg_var_cnt[0] <= 1e5: # and new_score <= -1e-5
        #     self.express_var_list.append(add_pat)
        #     logging.info('perfect separation pat found add to express')
        if new_score < 0:
            self.express_var_list.append(add_pat)
            # self.neg_var_cnt[0] += 1

        # if better, update best score so far

        if new_score < min_redu_cost[0]:
            min_redu_cost[0] = new_score
            # update the best_var_list
            while self.best_var_list:
                self.best_var_list.pop()
            self.best_var_list.append(add_pat)
            logging.info('better cost found, set cur pat as best {}'.format(self.best_var_list))
            return True
        elif new_score == min_redu_cost[0]:
            self.best_var_list.append(add_pat)
            logging.info('equal cost found, add cur pat to best')
            return False
        else:
            logging.info('worse cost found, not add cur pat')
            return False

    # fun to construct a formatted pattern to pool
    def generate_candi_format(self, pat, cnt, pos_bit, neg_bit, score):
        res = {}
        res['pattern'] = pat
        res['out_cnt'] = cnt
        res['pos_bit'] = pos_bit
        res['neg_bit'] = neg_bit
        res['score'] = score
        res['pos_rule_coef'] = len(pos_bit)
        res['neg_rule_coef'] = len(neg_bit)
        if cover_index_tobit:
            add_all_bit = pos_bit.union(neg_bit)
            overlap_cover_coef = len(add_all_bit.intersection(cover_index_tobit))
        else:
            overlap_cover_coef = 0
        res['overlap_cover_coef'] = overlap_cover_coef
        res['pat_len'] = len(pat)
        return res

    # fun to update cond node info's component succintly
    def update_single_info(self, node_info, comp_name, comp_add):
        if not node_info[comp_name]:
            node_info[comp_name] = comp_add
        else:
            node_info[comp_name] = node_info[comp_name].intersection(comp_add)


# main optimization function
def solve_mip(model, new_vars, relax_ind, pos_beta, overlap_beta, write_model):
    start_time = time.time()
    update_rules = col_list + new_vars
    cl_res_pos = copy.deepcopy(update_rules)
    # cl_res_pos = copy.deepcopy(new_vars)
    logging.info('MIP tot rule vars {}'.format(len(cl_res_pos)))

    n_pos = len(full_bit_dict['tot_pos_bit'])
    nx = len(cl_res_pos)

    # dual var constraint index
    mu_index = list(np.array(range(n_pos)))
    # lambda_index = [n_pos]

    m = Model()
    m.clear()
    # add y var repr miss pos data
    y = [m.add_var(name='y' + str(i), var_type=CONTINUOUS, lb=0, obj=pos_beta + overlap_beta) for i in range(n_pos)]
    # v = [m.add_var(name='v'+str(i), var_type=CONTINUOUS, lb=0, obj=overlap_beta) for i in range(n_pos)]
    for i in range(n_pos):
        # m += y[i] - v[i] == 1
        m += y[i] >= 1

    # add vars column-wise
    constrs_ = [[m.constrs[j] for j in cl_res_pos[i]['pos_bit']] for i in range(nx)]
    coeffs_ = [np.ones(len(cl_res_pos[i]['pos_bit'])) for i in range(nx)]
    objs_ = []
    for i in range(nx):
        objs_i = cl_res_pos[i]['pos_rule_coef'] * overlap_beta + cl_res_pos[i]['neg_rule_coef'] + rule_cost_coef * \
                 cl_res_pos[i]['pat_len']
        objs_i += cl_res_pos[i]['overlap_cover_coef'] * overlap_beta
        objs_.append(objs_i)
    prev_add_new_num = 0
    # prev_add_new_num = len(m.vars) - n_pos
    x = [m.add_var(name='x' + str(prev_add_new_num + i), var_type=CONTINUOUS, lb=0, obj=objs_[i],
                   column=Column(constrs=constrs_[i], coeffs=coeffs_[i])) for i in range(nx)]
    # for i in range(nx):
    #     logging.debug('add var {} with info {}'.format('x'+str(prev_add_new_num+i), cl_res_pos[i]))

    construct_time = time.time() - start_time

    m.verbose = False
    start = time.time()

    # m.lp_method = 1
    # print('method {}'.format(m.lp_method))
    m.optimize(max_seconds=60, relax=relax_ind)
    mip_time = time.time() - start
    logging.debug('is relax? {}'.format(relax_ind))
    mu_array = np.array([Constr(m, i).pi for i in mu_index]).copy()

    iteration[0] += 1
    if write_model:
        m.write('model/model' + str(iteration[0]) + '.lp')
        m.write('model/model' + str(iteration[0]) + '.sol')

    tot_var = len(m.vars)
    # new_tot = col_var+[i['pattern'] for i in new_vars]
    # print(tot_var, len(new_tot))
    mutiplier = 1
    if relax_ind:
        good_list = [(i, m.vars[i].x) for i in range(tot_var) if m.vars[i].x > 0 and i >= mutiplier * n_pos]
    else:
        good_list = [(i, m.vars[i].x) for i in range(tot_var) if m.vars[i].x == 1 and i >= mutiplier * n_pos]
        logging.debug([m.vars[i].x for i in range(tot_var)])
        logging.debug([(i, col_var[i]) for i in range(tot_var) if m.vars[i].x == 1 and i >= mutiplier * n_pos])
    # print(good_list)
    short_rule = [col_var[i[0] - mutiplier * n_pos] for i in good_list]
    # print(short_rule)

    logging.debug('this many constrs {}'.format(len(m.constrs)))
    basic_list = [(i, m.vars[i].x) for i in range(tot_var) if m.vars[i].x > 0]
    logging.debug('this many feasible basic solutions {}'.format(len(basic_list)))

    # lambda_array = np.array([Constr(m, i).pi for i in lambda_index])
    # lambda_array = [0.0]
    logging.debug('min mu array is {}'.format(min(mu_array)))
    # logging.info(lambda_array)
    # logging.info('ny {} nx {}'.format(ny, nx))
    # logging.info('obj value is {}'.format(m.objective))
    logging.info('obj value is {}'.format(m.objective_value))

    return {'short_rule': short_rule, 'mip_time': round(mip_time, 3),
            'm_object': m, 'mu_array': mu_array, 'construct_time': construct_time,
            'short_rule_index': [i[0] - mutiplier * n_pos for i in good_list]
            }


# function to estimate redu cost lower bound
def tree_redu_cost(neg_cnt_lb, pos_cnt_lb, overlap_beta, cond_node_pos_bit, cond_node_len, bound_type, fun_type):
    # longest path lead to lower bound neg count
    cost_part1 = neg_cnt_lb
    cost_part2 = pos_cnt_lb * overlap_beta

    test_rule_pos_index = cond_node_pos_bit
    # if fun_type == 'bound':
    #     # cost_part2 = mu_array_pos[test_rule_pos_index.to_array()].sum()
    #     cost_part2 = max_weight_dual
    #     new_pos_dual_interbit = cond_node_pos_bit.intersection(mu_array_pos_index)
    #     min_pos_weight_dual =  mu_array[new_pos_dual_interbit.to_array()].sum()
    #     cost_part2 += min_pos_weight_dual
    # else:
    #     cost_part2 = mu_array[test_rule_pos_index.to_array()].sum()
    cost_part3 = mu_array[test_rule_pos_index.to_array()].sum()

    cost_part4 = rule_cost_coef * cond_node_len
    # if bound_type == 'single':
    #     tot_cost = cost_part1 + cost_part2 - cost_part3 + cost_part4
    # else:
    #     tot_cost =  cost_part1 + cost_part2 - cost_part3 + cost_part4
    tot_cost = cost_part1 + cost_part2 - cost_part3 + cost_part4
    logging.debug('current tree reduced cost {} is {}'.format(bound_type, round(tot_cost, 3)))
    return tot_cost

