"""
MIT drillUp algorithm for pattern detection
Author: Kai He <kai.he@alibaba-inc.com>
Fan Yang <fanyang.yf@alibaba-inc.com>
Cheng Zhou <zhoucheng.zc@alibaba-inc.com>
"""
import re

import pandas as pd
import logging
from multiprocessing import Pool
from pyroaring import BitMap
import numpy as np
import time
import ast
from mindxlib.base.explainer import RuleExplainer
from mindxlib.base.explanation import RuleExplanation


class DrilluptExplanation(RuleExplanation):
    def __init__(self, rules, default_rule=0):
        self.rules = rules  # 规则列表，每个规则是一个包含多个 'feature:value' 字符串的列表
        self.default_rule = default_rule  # 默认规则

    def show(self):
        """Override show method to print rules in custom format."""
        N = len(self.rules)
        if N > 0:
            # 输出第一个规则
            print(f"IF {self._format_conditions(self.rules[0])}, THEN 1")
            # 输出其余的规则
            for ii in range(1, N):
                print(f"ELIF {self._format_conditions(self.rules[ii])}, THEN 1")
            # 输出默认规则
            print("ELSE 0")
        else:
            # 如果没有规则，仅输出默认规则
            print(f"IF THEN {self.default_rule}")

    def _format_conditions(self, conditions):
        """
        Helper method to format the conditions of a rule into a readable string.
        
        Parameters:
            conditions (list of str): List of 'feature:value' strings.
            
        Returns:
            str: Formatted condition string.
        """
        formatted_conditions = []
        for condition in conditions:
            feature, value = condition.split(':')
            formatted_conditions.append(f"{feature}=={value}")
        return " AND ".join(formatted_conditions)


logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
# logging.getLogger().setLevel(logging.DEBUG)
# logging.getLogger().setLevel(logging.INFO)

# preprocess: detect col that has only one value or two many values
def detect_bad_cols(df, dim_list):
    """
    Parameters
        df: pandas df, original data
        dim_list: user pre-specified columns, may contain bad ones
    Return
        list, list of col names to exclude from dim_list
    """
    bad_cols = []
    num = df.shape[0]
    for col in dim_list:
        num_lvls = df[col].nunique(dropna=True)
        if num_lvls <= 1 or num_lvls/num > 0.7:
            bad_cols.append(col)
        else:
            max_cnt = max(df[col].dropna().value_counts().values)
            max_ratio = max_cnt/num
            if max_ratio > 0.95:
                bad_cols.append(col)


    return bad_cols


# fill missing values
def fill_miss(df, dim_list):
    """
    Parameters
        df: pandas df
        dim_list: dimension list
    Return
        a filled missing value df, if a dim is numeric
        set to str and fill with 'unknown<'
    """
    miss_cols = df[dim_list].columns[df[dim_list].isnull().any()].tolist()
    fill_dict = {}
    for col in dim_list:
        if df[col].dtype != object:
            df[col] = df[col].astype(str)
        if col in miss_cols:
            fill_dict[col] = 'unknown<'
    return df.fillna(fill_dict)


# reduce long tail dim_val, replace with 'other<'
def reduce_dim_val(df, dim_list, min_dim_val_cnt):
    """
    Parameters
        df: dataframe with missing value filled
        dim_list: dimension list
        min_dim_val_cnt: min dim value count to be replaced with 'other<'
    Return
        a pandas dataframe, with "col_has_other_map_no_other" col
    """
    add_col = []
    for col in dim_list:
        # val_cnt = df[col].value_counts()
        val_cnt = df[[col, 'count']].groupby([col]).agg('sum').reset_index()
        other_candi = list(val_cnt[val_cnt['count'] < min_dim_val_cnt][col].values)
        if len(other_candi) > 10:
            logging.info("col {} needs replace dim val".format(col))
            df.loc[df[col].isin(other_candi), col] = 'other<'

            new_dim_val = df[col].unique()
            no_other_candi = [i for i in new_dim_val if i != 'other<']
            add_dict = {col: no_other_candi}
            add_col.append(str(add_dict))
    logging.info("reduce dim is done!")
    return df, add_col


# preprocess: group by dim list to create a compressed version for fp
def compress_data(df, dim_list, label_col, label_val):
    """
    Parameters
        df: pandas df, original data
        dim_list: cleaned dim list for group by use
    Return
        pandas df, add a 'count' column
    """
    df = df[dim_list+[label_col, 'count']]
    groupby_df = df.groupby(dim_list+[label_col]).agg('sum').reset_index()
    # label_index = BitMap(groupby_df[groupby_df[label_col] == label_val].index)
    return groupby_df


# core drill up procedure
def preprocess(df, label_col, label_val, dim_list, min_dim_val_cnt):
    """
    Parameters
        df: dataframe after preprocessed, groupby, add 'count' col
        label_col: column name to represent label;
        label_val: label_col == label_val -> outlier group
        dim_list: list, cols used to drill up
        min_dim_val_cnt: min dim value count to be replaced with 'other<'
    Return
        a pandas dataframe
    """

    # step 0.b check label_col and label_val type
    if df[label_col].dtype == object and type(label_val) != str:
        label_val = str(label_val)
    if df[label_col].dtype == int and type(label_val) != int:
        label_val = int(float(label_val))

    df['count'] = 1

    # step 0.c detect bad columns, remove from dim list

    bad_col = detect_bad_cols(df, dim_list)
    logging.info("Bad cols are {}".format(','.join(str(bad_col))))
    dim_list = [i for i in dim_list if i not in bad_col]
    logging.info("After bad_col detect, dim_list is {}".format(
                 ','.join(str(dim_list))))

    if len(dim_list) < 1:
        raise ValueError("All columns have been detected as bad columns.")
    
    df = fill_miss(df, dim_list)
    logging.info("fill missing values!")

    df, map_col = reduce_dim_val(df, dim_list, min_dim_val_cnt)

    # step 0.d compress data
    c_df = compress_data(df, dim_list, label_col, label_val)
    logging.info("After compress, data shape is ({},{})".format(
                 c_df.shape[0], c_df.shape[1]
                 ))
    c_out = c_df[c_df[label_col] == label_val]
    c_in = c_df[c_df[label_col] != label_val]
    if c_out['count'].sum()/c_df['count'].sum() < 0.001:
        c_in_random = random.sample(list(c_in.index), 1000*c_out['count'].sum())
        remain_index = list(c_out.index) + c_in_random
        c_df = c_df.loc[remain_index, :].reset_index(drop=True)
        logging.info("Huge imbalance, truncate data to ({},{})".format(
                 c_df.shape[0], c_df.shape[1]))

    c_df['col_has_other_map_no_other'] = ''
    c_df.loc[range(len(map_col)), 'col_has_other_map_no_other'] = map_col
    c_df['dim_list'] = ''
    c_df.loc[0, 'dim_list'] = str(dim_list)

    return c_df, dim_list

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
    col_val = pd.DataFrame(
        {col: str(col) + ":" for col in df[dim_list]}, index=df[dim_list].index
    ) + df[dim_list].astype(str)

    return col_val.values.tolist()


# create one-hot format of original data, save the bitMap rep in dict
def bitify_data(df, dim_list):
    """
    Parameters
        df: data to be bitify
        dim_list: list, cols used to drill up
    Return
        dict, each key:value is a col and its bit result in dim_list
    """
    full_bit_dict = {}
    if type(df) == pd.core.frame.DataFrame:
        data_encode = pd.get_dummies(df[dim_list], prefix_sep=':')
        for i in data_encode.columns:
            full_bit_dict[i] = BitMap(np.array(data_encode[i]).nonzero()[0])
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
    item_list = list(pattern)
    # get count by use bit operation on full data
    filter_bit_list = []
    for i in range(len(item_list)):
        # print(item_list[i])
        filter_bit_list.append(data_bit_dict[item_list[i]])
        # print(data_bit_dict[item_list[i]])
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
        full_bit_dict, score_type, flip_sign) = input_tuple

    # get count by use bit operation on full data
    pattern_bit = filter_pat_via_bit(pattern, full_bit_dict)
    filter_df_len = get_pat_cnt(pattern_bit, full_bit_dict)
    # out1_len: not-pattern cnt in outlier
    # in0_len: pattern cnt in inlier
    # in1_len: not pattern cnt in inlier
    out1_len = out_cnt - pattern_cnt
    in0_len = filter_df_len - pattern_cnt
    in1_len = tot_cnt - out_cnt - in0_len
    # logging.info('details ({},{},{},{})'.format(pattern_cnt, in0_len, out1_len, in1_len))
    # support two score types
    if score_type == 'risk':
        comp1 = pattern_cnt/(pattern_cnt+in0_len)
        comp2 = (out1_len+1e-4)/(out1_len+in1_len+1e-4)
        score = comp1/comp2
    elif score_type == 'diffScore':
        recall = pattern_cnt/out_cnt
        # logging.info('pattern_cnt {} out_cnt {}'.format(pattern_cnt, out_cnt))
        precision = pattern_cnt/filter_df_len if filter_df_len else 0
        score = 2*precision*recall/(precision+recall)
        # logging.info('precision {} recall {}'.format(precision, recall))
    elif score_type == 'rep':
        score = (pattern_cnt - in0_len)/filter_df_len
    elif score_type == 'rep*':
        score1 = (pattern_cnt - in0_len)/filter_df_len
        comp1 = pattern_cnt/(pattern_cnt+in0_len)
        comp2 = (out1_len+1e-4)/(out1_len+in1_len+1e-4)
        score2 = comp1/comp2
        score = score1*score2

    details = (round(pattern_cnt, 2), round(in0_len, 2),
               round(out1_len, 2), round(in1_len, 2))

    if flip_sign:
        score = -score

    return [pattern, pattern_cnt, details, round(score, 4)]


# multi-process to evalute score each pattern candidate
def score_candi(itemset_candi, out_cnt, tot_cnt, full_bit_dict, score_type,
                flip_sign):
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
    input_tuple_list = [(i[0], i[1], out_cnt, tot_cnt, full_bit_dict,
                        score_type, flip_sign) for i in itemset_candi]
    with Pool(16) as p:
        new_res_bit = p.map(mp_f_bit, input_tuple_list)

    return sorted(new_res_bit, key=lambda x: x[-1], reverse=True)


# use bitmap to compute jaccard similarity
def jcd_dist(pattern1, pattern2, full_bit_dict):
    """
    Parameters
        pattern1, pattern2: patterns to compare
        full_bit_dict: dict for full data, with col as key, value is bitMap
    Return
        float, len of intersect data/union by two patterns
    """
    filter_bit1 = filter_pat_via_bit(pattern1, full_bit_dict)
    filter_bit2 = filter_pat_via_bit(pattern2, full_bit_dict)
    intersect = filter_bit1.intersection(filter_bit2)
    intersect_len = get_pat_cnt(intersect, full_bit_dict)

    union = filter_bit1.union(filter_bit2)
    union_len = get_pat_cnt(union, full_bit_dict)
    return intersect_len/union_len


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
        label_bit = full_bit_dict['label_index']
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
    logging.info('tot matched label {}'.format(tot_matched_label))
    return out_df


# sort and build col_to_index + index_to_col dict
def build_ref_dict(full_bit_dict):
    """
    Parameters
        full_bit_dict: dict for full data, with col as key, value is bitMap
    Return
        dict, dict, column to index map, and vice verse
    """
    sorted_keys = sorted(full_bit_dict)
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


# core drill up procedure
def drillUp(c_df, label_col, label_val, col_cond_dict, dim_list,
            min_dim_val_cnt, sup_ratio, out_num, jcd_limit,
            min_pat_len, score_type, score_gap, test_dim_num):
    """
    Parameters
        c_df: dataframe after preprocessed, groupby, add 'count' col
        label_col: column name to represent label;
        label_val: label_col == label_val -> outlier group
        col_cond_dict: dict, if label_col not create yet, use this to create
        dim_list: list, cols used to drill up
        min_dim_val_cnt: min dim value count to be replaced with 'other<'
        sup_ratio: support ratio
        out_num: number of ranked top results
        jcd_limit: float, jaccard similirity, add new pattern when < limit
        min_pat_len: minimum len of pattern to include, to prune bad case
        score_type: 'risk' and 'diffScore'
    Return
        a pandas dataframe
    """
    c_df['count'] = c_df['count'].astype(float)
    if dim_list is None:
        dim_list = ast.literal_eval(c_df['dim_list'].values[0])
    if test_dim_num:
        new_num = min(test_dim_num, len(dim_list))
        dim_list = dim_list[:new_num]
    # step 1. split the data
    outlier_df, inlier_df = split_data(c_df, label_col, label_val)
    logging.info("Data split finished!")
    logging.info("Ouliter shape {} and cnt {}".format(outlier_df.shape[0],
                 outlier_df['count'].sum()))

    # step 2. get bit format for dataframe
    outlier_bit_dict = bitify_data(outlier_df, dim_list)
    outlier_bit_dict_key = list(outlier_bit_dict.keys()).copy()
    outlier_bit_dict['count'] = np.array(outlier_df['count'])
    # initial item-count dict for algo
    init_dict = {}
    for key in outlier_bit_dict_key:
        # print(key)
        bit_to_array = outlier_bit_dict[key].to_array()
        init_dict[key] = outlier_bit_dict['count'][bit_to_array].sum()

    full_bit_dict = bitify_data(c_df, dim_list)
    full_bit_dict['count'] = np.array(c_df['count'])
    if 'contr' in c_df.columns:
        full_bit_dict['contr'] = np.array(c_df['contr'])
    full_bit_dict['tot_bit'] = BitMap(list(c_df.index))
    label_index = BitMap(c_df[c_df[label_col] == label_val].index)
    full_bit_dict['label_index'] = label_index
    logging.info("Data to bitmap finished! tot keys {}".format(
                 len(full_bit_dict)))
    col_to_index, index_to_col = build_ref_dict(full_bit_dict)

    # step 3. fp-growth for both oulier and inlier
    outlier_list = get_col_val_itemset(outlier_df, dim_list)
    outlier_weight = list(outlier_df['count'].values)
    outlier_min_sup = outlier_df['count'].sum() * sup_ratio

    out_cnt = outlier_df['count'].sum()
    tot_cnt = c_df['count'].sum()

    outlier_fp_tree = cl_fptree(outlier_list, outlier_weight, {}, [], None,
                                outlier_min_sup,
                                out_cnt, tot_cnt, full_bit_dict, init_dict,
                                col_to_index, index_to_col, score_type,
                                [-float("inf")], score_gap, {}, False)
    outlier_fp_tree.findfqt()
    frequentitemset = outlier_fp_tree.get_closed_list()
    frequentitemset = sorted(frequentitemset, key=lambda k: -k[1])
    logging.info("Main pattern algorithm is done!")
    logging.info("Total candis {}".format(len(frequentitemset)))
    # logging.info('initial drill result {}'.format(frequentitemset))

    # step 4: compute risk ratios
    cl_res = score_candi(frequentitemset, out_cnt, tot_cnt, full_bit_dict,
                         score_type, False)  # [pattern, pattern_cnt, interestingness_score] pattern_cnt从outlier计算得来
    logging.info("All patterns are scored!")
    # step 5: post process and results out
    cl_slim_res = post_process(cl_res, out_num, jcd_limit, min_pat_len,
                               full_bit_dict)
    # logging.info('slim result flow into post process {}'.format(cl_slim_res))
    out_df = final_res_assemble(cl_slim_res, out_num, label_col, label_val,
                                full_bit_dict)  # [rules, count, label_col + "_ratio"'
    other_map_info = pd.DataFrame(c_df['col_has_other_map_no_other'])
    other_map_info = other_map_info[other_map_info.values != '']
    other_map_info = other_map_info[~(other_map_info['col_has_other_map_no_other'].isnull())]
    logging.info("Final result is finished!")
    if type(out_df) is list:
        return out_df

    out_df['link'] = range(out_df.shape[0])
    other_map_info['link'] = range(other_map_info.shape[0])
    return_df = out_df.merge(other_map_info, on=['link'], how='outer')
    return_df = return_df[[i for i in return_df.columns if i != 'link']]
    return return_df, cl_res, cl_slim_res, outlier_fp_tree

    # if out_df.shape[0] < other_map_info.shape[0]:
    #     other_map_info['rules'] = ''
    #     other_map_info.loc[range(out_df.shape[0]), 'rules'] = \
    #         out_df["rules"].values
    #     return other_map_info
    # else:
    #     out_df['col_has_other_map_no_other'] = ''
    #     out_df.loc[range(other_map_info.shape[0]),
    #                'col_has_other_map_no_other'] = \
    #         other_map_info["col_has_other_map_no_other"].values
    #     return out_df


# compute pattern upper bound info
def get_score_ub(pattern_cnt, pat_lb, out_cnt, tot_cnt, score_type, flip_sign):
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
        comp1 = pattern_cnt/(pattern_cnt+in0_len)
        comp2 = (out1_len+1e-4)/(out1_len+in1_len+1e-4)
        score = comp1/comp2
    elif score_type == 'diffScore':
        recall = pattern_cnt/out_cnt
        precision = pattern_cnt/(pattern_cnt+in0_len)
        score = 2*precision*recall/(precision+recall)
    elif score_type == 'rep':
        score = (pattern_cnt-in0_len)/(pattern_cnt+in0_len)
    elif score_type == 'rep*':
        score1 = (pattern_cnt-in0_len)/(pattern_cnt+in0_len)
        comp1 = pattern_cnt/(pattern_cnt+in0_len)
        comp2 = (out1_len+1e-4)/(out1_len+in1_len+1e-4)
        score2 = comp1/comp2
        score = score1*score2

    if flip_sign:
        score = -score

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
        self.bit = []


class cl_fptree:
    def __init__(self, data, freq_weight, fc_list, cond_node_list,
                 cond_node_bit, minsup, out_cnt, tot_cnt, full_bit_dict,
                 itm_dict_in, col_to_index, index_to_col, score_type,
                 max_score, score_gap, time_profile, flip_sign):
        # raw data and minminual support
        self.data = data
        self.minsup = minsup
        self.freq_weight = freq_weight
        self.fc_list = fc_list
        self.cond_node_list = cond_node_list
        self.cond_node_bit = cond_node_bit
        self.out_cnt = out_cnt
        self.tot_cnt = tot_cnt
        self.full_bit_dict = full_bit_dict
        self.itm_dict_in = itm_dict_in
        self.col_to_index = col_to_index
        self.index_to_col = index_to_col
        self.score_type = score_type
        self.max_score = max_score
        self.score_gap = score_gap
        self.flip_sign = flip_sign
        if not time_profile:
            self.time_profile = {
                                 'construct': {'time': 0.0, 'cnt': 0},
                                 'condtree': {'time': 0.0, 'cnt': 0},
                                 'condtree_ub': {'time': 0.0, 'cnt': 0},
                                 'prune': {'cnt': 0, 'volume': 0}
                                }
        else:
            self.time_profile = time_profile

        # null root
        self.root = cl_node("Null", 1.0, None, None)
        self.root.bit = self.full_bit_dict['tot_bit']

        # each line of transaction with new order from the most
        # frequent items to less
        self.itm_line_sort = []
        # node table containing link of all nodes of same item
        self.node_table = []
        # dictionary contaiing item more than the minsupport count
        # with des order
        self.itm_sort_dict = []
        # dictionaly containing item and the support count
        self.itm_dict = {}
        # dictionary with item and it's postion of the support count rank
        self.itm_order_dict = {}

        self.construct(data)
        # second scan and build fp tree line  by line

    def update_cond_node_bit(self, new_col):
        # logging.info('updating cond node bit {}'.format(new_col))
        new_col_bit = self.full_bit_dict[new_col]
        if not self.cond_node_bit:
            self.cond_node_bit = new_col_bit
        else:
            self.cond_node_bit = self.cond_node_bit.intersection(new_col_bit)
        # logging.info('after updating, cond node bit is {}'.format(
        #              self.cond_node_bit))

    def construct(self, data):
        logging.debug('building...')
        start_time = time.time()
        if not self.itm_dict_in:
            # get support count for all item
            for k, tran in enumerate(data):
                for itm in tran:
                    if itm in self.itm_dict.keys():
                        self.itm_dict[itm] += self.freq_weight[k]
                    else:
                        self.itm_dict[itm] = self.freq_weight[k]
        else:
            self.itm_dict = self.itm_dict_in.copy()

        # extract pattern in every trans
        if not self.cond_node_list:
            cond_node_cnt = sum(self.freq_weight)
        else:
            cond_node_cnt = min([i["itemcc"] for i in self.cond_node_list])
        extract_list = []
        for each_item in self.itm_dict:
            this_item_cnt = self.itm_dict[each_item]
            if this_item_cnt == cond_node_cnt:
                logging.debug('found same cnt item cnt {} and cond node cnt \
                             {}'.format(this_item_cnt, cond_node_cnt))
                extract_list.append(each_item)
        if extract_list:
            logging.debug('found extract list {}'.format(extract_list))
            if not self.cond_node_list:
                self.add_a_pattern([set(extract_list), cond_node_cnt])
            else:
                add_set = set(extract_list + [i["itemn"]
                              for i in self.cond_node_list])
                self.add_a_pattern([add_set, cond_node_cnt])
            for item_i in extract_list:
                self.itm_dict.pop(item_i)
                self.cond_node_list.append({"itemn": item_i,
                                            "itemcc": cond_node_cnt})
                self.update_cond_node_bit(item_i)
        else:
            if self.cond_node_list:
                logging.debug('not found extract list, register cond node')
                add_set = set([i["itemn"] for i in self.cond_node_list])
                self.add_a_pattern([add_set, cond_node_cnt])

        itemlist = list(self.itm_dict.keys())
        # prune all the world with < min support count
        for itm in itemlist:
            if self.itm_dict[itm] < self.minsup:
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
            t += 1
            iteminfo = {"itemn": item_n, "itemcc": item_c, "linknode": None}
            self.node_table.append(iteminfo)

        # construct fptree line by line
        for k, line in enumerate(data):
            supitem = []
            for itm in line:
                # only keep items with support count higher than minsupport
                if itm in self.itm_dict.keys():
                    supitem.append(itm)
            # insert items to the fp tree
            if len(supitem) > 0:
                # reorder the items
                sortsupitem = sorted(supitem,
                                     key=lambda k: self.itm_order_dict[k])
                self.itm_line_sort.append(sortsupitem)
                # enter the item one by one from begining
                R = self.root
                for i in sortsupitem:
                    if i in R.children.keys():
                        R.children[i].itm_count += self.freq_weight[k]
                        R = R.children[i]
                    else:

                        R.children[i] = cl_node(
                            i, self.freq_weight[k], R, None
                        )
                        R.children[i].bit = self.full_bit_dict[i].intersection(R.bit)
                        R = R.children[i]
                        # link this node to node_table
                        for iteminfo in self.node_table:
                            if iteminfo["itemn"] == R.itm:
                                # find the last node of the  node linklist
                                if iteminfo["linknode"] is None:
                                    iteminfo["linknode"] = R
                                else:
                                    iter_node = iteminfo["linknode"]
                                    while iter_node.link is not None:
                                        iter_node = iter_node.link
                                    iter_node.link = R
        self.dir_get_fc()
        # self.time_profile['construct']['cnt'] += 1
        # self.time_profile['construct']['time'] += time.time() - start_time

    # create transactions for conditinal tree
    def condtreetran(self, N):
        if not self.cond_node_list:
            self.cond_node_bit = None
        else:
            first_cond_node = self.cond_node_list[0]['itemn']
            self.cond_node_bit = self.full_bit_dict[first_cond_node]
            for i in self.cond_node_list[1:]:
                new_item_bit = self.full_bit_dict[i['itemn']]
                update_bit = self.cond_node_bit.intersection(new_item_bit)
                self.cond_node_bit = update_bit
        # uncomment for debug
        logging.debug('working on cond tree on {}'.format(N.itm))
        logging.debug('current cond node list {}'.format([i['itemn'] for i in
                      self.cond_node_list]))
        # logging.info('current cond node bit {}'.format(self.cond_node_bit))
        start_time = time.time()
        if N.parent is None:
            return None

        condtreeline = []
        cond_freq_weight = []
        cond_dict = {}
        min_sup = float("inf")
        # starting from the leaf node reverse add item till hit root
        ub_time = 0.0
        ub_cnt = 0
        # score_ub = -float("inf")
        itm_link_list = []
        # tot_line_list = []
        while N is not None:
            line = []
            PN = N.parent
            # line_bit_list = [self.full_bit_dict[N.itm]]
            name_list = [N.itm]
            while PN.parent is not None:
                line.append(PN.itm)
                # line_bit_list.append(self.full_bit_dict[PN.itm])
                name_list.append(PN.itm)
                cur_cnt = cond_dict[PN.itm] if PN.itm in cond_dict else 0
                cond_dict[PN.itm] = cur_cnt + N.itm_count
                PN = PN.parent
            line = line[::-1]
            condtreeline.append(line)
            cond_freq_weight.append(N.itm_count)

            ub_start_time = time.time()
            # min_pat = line + [N.itm]
            # new_bit = filter_pat_via_bit(min_pat, self.full_bit_dict)
            line_bit_list = [N.bit]
            if self.cond_node_bit:
                line_bit_list.append(self.cond_node_bit)
            # tot_line_list.append((line_bit_list, N.itm_count))
            # print('line info is', line)
            min_pat_lb_bit = BitMap.intersection(*line_bit_list)
            min_pat_lb_cnt = get_pat_cnt(min_pat_lb_bit, self.full_bit_dict)
            # print('algo found tot line cnt', min_pat_lb_cnt)
            min_pat_lb_cnt = min_pat_lb_cnt - N.itm_count
            # print('compute line cnt in inlier', min_pat_lb_cnt)

            if min_sup > min_pat_lb_cnt:
                min_sup = min_pat_lb_cnt
            ub_time += time.time() - ub_start_time
            ub_cnt += 1
            itm_link_list.append(N.itm_count)

            N = N.link

        # for i in itm_link_list:
        #     new_ub = get_score_ub(i, min_sup, self.out_cnt, self.tot_cnt)
        #     if new_ub > score_ub:
        #         score_ub = new_ub
        score_ub = get_score_ub(sum(itm_link_list), min_sup, self.out_cnt,
                                self.tot_cnt, self.score_type, self.flip_sign)
        if sum(itm_link_list) < min_sup:
            score_ub = -float('inf')

        # self.time_profile['condtree']['cnt'] += 1
        # self.time_profile['condtree']['time'] += time.time() - start_time
        # self.time_profile['condtree_ub']['time'] += ub_time
        # self.time_profile['condtree_ub']['cnt'] += ub_cnt

        return condtreeline, cond_freq_weight, min_sup, cond_dict, score_ub

    # Find frequent item list by creating conditional tree
    def findfqt(self, parentnode=None):
        if len(list(self.root.children.keys())) == 0:
            return None
        result = []
        sup = self.minsup
        # starting from the end of node_table
        revtable = self.node_table[::-1]
        logging.debug('node table order {}'.format(revtable))
        for n in revtable:
            logging.debug('analyzing node {}'.format(n['itemn']))
            fqset = [set(), 0]
            if parentnode is None:
                fqset[0] = {
                    n["itemn"],
                }
            else:
                fqset[0] = {n["itemn"]}.union(parentnode[0])
            fqset[1] = n["itemcc"]
            if fqset[1] >= self.minsup:
                result.append(fqset)
            condtran, condtran_freq_weight, candi_lb, cond_dict, score_ub = \
                self.condtreetran(n["linknode"])
            # recursively build the conditinal fp tree
            add_n = n.copy()
            add_n.pop("linknode")
            new_cond_node_list = self.cond_node_list + [add_n]
            # uncomment to debug
            new_pattern = [i['itemn'] for i in new_cond_node_list]
            logging.debug("condtree {} upper bound {}, best {}".format(
                         ','.join(new_pattern), score_ub, self.max_score[0]))
            if score_ub > self.score_gap * self.max_score[0]:
                self.update_cond_node_bit(add_n['itemn'])
                contree = cl_fptree(
                    condtran,
                    condtran_freq_weight,
                    self.fc_list,
                    new_cond_node_list.copy(),
                    self.cond_node_bit.copy(),
                    sup,
                    self.out_cnt,
                    self.tot_cnt,
                    self.full_bit_dict,
                    cond_dict.copy(),
                    self.col_to_index,
                    self.index_to_col,
                    self.score_type,
                    self.max_score,
                    self.score_gap,
                    self.time_profile,
                    self.flip_sign
                )
                conitems = contree.findfqt(fqset)
            # else:
                # self.time_profile['prune']['cnt'] += 1
                # self.time_profile['prune']['volume'] += len(condtran)
                # logging.info('placeholer')
        return

    def get_closed_list(self):

        res = []
        for key in self.fc_list:
            # add_list = []
            reverse_pat_len_list = sorted(self.fc_list[key].items(), key=lambda x:x[0], reverse=True)
            slim_add_list = []
            for each_candi in reverse_pat_len_list:
                if not slim_add_list:
                    slim_add_list += each_candi[1]
                else:
                    for each_test_candi in each_candi[1]:
                        add_test = True
                        for each_exist_candi in slim_add_list:
                            if each_test_candi.issubset(each_exist_candi):
                                add_test = False
                                break
                        if add_test:
                            slim_add_list.append(each_test_candi)

            # for key1 in self.fc_list[key]:
            #     add_list += self.fc_list[key][key1]

            res += [[[self.index_to_col[j] for j in i], key] for i in slim_add_list]
        logging.info('time_profile {}'.format(self.time_profile))
        return res

    def dir_get_fc(self):
        for node in self.node_table:
            # check how many times this node appear
            linknode = node["linknode"]
            count = 1
            while linknode.link is not None:
                linknode = linknode.link
                count += 1
            if count == 1:
                #                 print('single node found', node)
                continue

    def add_a_pattern(self, i):

        my_dict = self.fc_list
        len_p = len(i[0])
        freq_p = round(i[1], 4)
        p = i[0]
        p_bit = pat_bit(p, self.col_to_index)

        global_pat_bit = filter_pat_via_bit(p, self.full_bit_dict)
        global_pat_len = get_pat_cnt(global_pat_bit, self.full_bit_dict)

        logging.debug('now try to add pattern {}'.format(i))
        logging.debug('with lengh {}, freq {}, global pat len \
                     {}'.format(len_p, freq_p, global_pat_len))

        if global_pat_len <= 1.05 * self.out_cnt and 2*freq_p > global_pat_len:
            logging.debug('passing add pattern requirement, \
                          few remaining check to go')
            add_ind = False

            if freq_p not in my_dict:
                logging.debug('cur freq not in pool, start new key:val pair')
                my_dict[freq_p] = {len_p: [p_bit]}
                add_ind = True
            else:
                # check if existing set has len longer than p:
                # possible closed compare
                logging.debug('cur freq in pool, check if > pat len key exists')
                target_len = [key for key in my_dict[freq_p] if key > len_p]
                # target_len = [key for key in my_dict[freq_p]]
                # add if no possible supersets exist
                if not target_len:
                    logging.debug('check pat > len not exists')
                    if len_p in my_dict[freq_p]:
                        logging.debug('cur freq exists, append')
                        my_dict[freq_p][len_p].append(p_bit)
                    else:
                        logging.debug('cur freq not exists, create new')
                        my_dict[freq_p][len_p] = [p_bit]
                    add_ind = True
                else:
                    logging.debug('check pat len exists, check if superset exists')
                    good_ind = True
                    for each_len in target_len:
                        for compare_p in my_dict[freq_p][each_len]:
                            if p_bit.issubset(compare_p):
                                logging.debug('super set found, not add pat')
                                good_ind = False
                                break
                        if not good_ind:
                            break
                    if good_ind:
                        add_ind = True
                        if len_p not in my_dict[freq_p]:
                            my_dict[freq_p][len_p] = [p_bit]
                        else:
                            my_dict[freq_p][len_p].append(p_bit)
            if add_ind:
                input_tuple = (p, freq_p, self.out_cnt, self.tot_cnt,
                               self.full_bit_dict, self.score_type, self.flip_sign)
                new_score = mp_f_bit(input_tuple)[-1]
                if self.max_score[0] < new_score:
                    tot_fc = len(self.fc_list)
                    logging.debug("add pat {}, tot {}".format(p, tot_fc))
                    logging.debug("update max score to {}".format(new_score))
                    self.max_score[0] = max(self.max_score[0], new_score)

        return

    def add_list_pattern(self, p_list):
        for p in p_list:
            self.add_a_pattern(p)

class DrillUp(RuleExplainer):
    def __init__(self,label_col="Label", label_val=None, dim_list=None,
                min_dim_val_cnt=5, sup_ratio=0.01, out_num=100, jcd_limit=0.75,
                min_pat_len=1, score_gap=1.0, score_type='risk'):
        """
        Parameters
            data_df: dataframe including features and label col
            label_col: column name to represent label;
            label_val: label_col == label_val -> outlier group
            dim_list: list, cols used to drill up
            min_dim_val_cnt: min dim value count to be replaced with 'other<'
            sup_ratio: support ratio
            out_num: number of ranked top results
            jcd_limit: float, jaccard similirity, add new pattern when < limit
            min_pat_len: minimum len of pattern to include, to prune bad case
            score_type: 'risk' and 'diffScore'
        """
        self.label_col = label_col
        self.label_val = label_val
        self.dim_list = dim_list
        self.min_dim_val_cnt = min_dim_val_cnt
        self.sup_ratio = sup_ratio
        self.out_num = out_num
        self.jcd_limit = jcd_limit
        self.min_pat_len= min_pat_len
        self.score_gap = score_gap
        self.score_type = score_type

    def fit(self, X, y, X_columns=None, y_column='Label',default_label=None):
        self.X_columns = X_columns
        X = self._ensure_dataframe(X,columns=self.X_columns)
        y = self._ensure_dataframe(y,columns=y_column if y_column else [self.label_col])

        
        label_counts = y.value_counts()
        if default_label is None:
            default_label = label_counts.idxmax()
            print(f"Using default rule name: {default_label} (most frequent class in data)")
        elif default_label not in list(label_counts.index):
            raise ValueError(f'default_label is not in the data: got {default_label}, expected one of {list(label_counts.index)}')
        self.default_label = default_label

        data_df = pd.concat((X,y),axis=1)
        if self.dim_list is None:
            self.dim_list = list(X.columns)
        
        c_df, self.dim_list= preprocess(data_df,self.label_col,self.label_val,self.dim_list,self.min_dim_val_cnt)

        c_df['count'] = c_df['count'].astype(float)

        # step 1. split the data
        outlier_df, inlier_df = split_data(c_df, self.label_col, self.label_val)
        logging.info("Data split finished!")
        logging.info("Ouliter shape {} and cnt {}".format(outlier_df.shape[0],
                                                          outlier_df['count'].sum()))

        # step 2. get bit format for dataframe
        outlier_bit_dict = bitify_data(outlier_df, self.dim_list)
        outlier_bit_dict_key = list(outlier_bit_dict.keys()).copy()
        outlier_bit_dict['count'] = np.array(outlier_df['count'])
        # initial item-count dict for algo
        init_dict = {}
        for key in outlier_bit_dict_key:
            # print(key)
            bit_to_array = outlier_bit_dict[key].to_array()
            init_dict[key] = outlier_bit_dict['count'][bit_to_array].sum()

        full_bit_dict = bitify_data(c_df, self.dim_list)
        full_bit_dict['count'] = np.array(c_df['count'])
        if 'contr' in c_df.columns:
            full_bit_dict['contr'] = np.array(c_df['contr'])
        full_bit_dict['tot_bit'] = BitMap(list(c_df.index))
        label_index = BitMap(c_df[c_df[self.label_col] == self.label_val].index)
        full_bit_dict['label_index'] = label_index
        logging.info("Data to bitmap finished! tot keys {}".format(
            len(full_bit_dict)))
        col_to_index, index_to_col = build_ref_dict(full_bit_dict)

        # step 3. fp-growth for both oulier and inlier
        outlier_list = get_col_val_itemset(outlier_df, self.dim_list)
        outlier_weight = list(outlier_df['count'].values)
        outlier_min_sup = outlier_df['count'].sum() * self.sup_ratio

        out_cnt = outlier_df['count'].sum()
        tot_cnt = c_df['count'].sum()

        outlier_fp_tree = cl_fptree(outlier_list, outlier_weight, {}, [], None,
                                    outlier_min_sup,
                                    out_cnt, tot_cnt, full_bit_dict, init_dict,
                                    col_to_index, index_to_col, self.score_type,
                                    [-float("inf")], self.score_gap, {}, False)
        outlier_fp_tree.findfqt()
        frequentitemset = outlier_fp_tree.get_closed_list()
        frequentitemset = sorted(frequentitemset, key=lambda k: -k[1])
        logging.info("Main pattern algorithm is done!")
        logging.info("Total candis {}".format(len(frequentitemset)))
        # logging.info('initial drill result {}'.format(frequentitemset))

        # step 4: compute risk ratios
        self.cl_res = score_candi(frequentitemset, out_cnt, tot_cnt, full_bit_dict,
                             self.score_type,
                             False)  # [pattern, pattern_cnt, interestingness_score] pattern_cnt从outlier计算得来
        logging.info("All patterns are scored!")
        # step 5: post process and results out
        self.cl_slim_res = post_process(self.cl_res, self.out_num, self.jcd_limit, self.min_pat_len,
                                   full_bit_dict)
        # logging.info('slim result flow into post process {}'.format(cl_slim_res))
        out_df = final_res_assemble(self.cl_slim_res, self.out_num, self.label_col, self.label_val,
                                    full_bit_dict)  # [rules, count, label_col + "_ratio"'
        other_map_info = pd.DataFrame(c_df['col_has_other_map_no_other'])
        other_map_info = other_map_info[other_map_info.values != '']
        other_map_info = other_map_info[~(other_map_info['col_has_other_map_no_other'].isnull())]
        logging.info("Final result is finished!")
        if type(out_df) is list:
            return out_df

        out_df['link'] = range(out_df.shape[0])
        other_map_info['link'] = range(other_map_info.shape[0])
        return_df = out_df.merge(other_map_info, on=['link'], how='outer')
        self.return_df = return_df[[i for i in return_df.columns if i != 'link']]
        self.output_rule = []
        for each in self.cl_slim_res:
            self.output_rule.append(each[0])
        self.rules = DrilluptExplanation(rules=self.output_rule, default_rule=self.default_label)
        return self

    def predict(self, X_test):
        X_test = self._ensure_dataframe(X_test, columns=self.X_columns)
       
        predictions = X_test.apply(self.rule_set_cover, rule_list=self.output_rule, axis=1)
        return predictions

    def rule_set_cover(self, row, rule_list):
        for i in range(len(rule_list)):
            tmprule = rule_list[i]
            if self.single_rule_cover(row, tmprule):  # 符合前提条件
                return 1
        return 0

    def single_rule_cover(self, row, rule):  # 判断一个样本是否服从rule的前提条件,rule is a list of items
        for eachfeature in rule:
            if ":" in eachfeature:
                feature_name = re.split(':', eachfeature)[0]
                feature_val = re.split(':', eachfeature)[1]
                if isinstance(row.index, pd.RangeIndex) or all(isinstance(idx, (int, np.integer)) for idx in row.index):
                    if str(row[int(feature_name)]) != feature_val:
                        return False
                elif all(isinstance(idx, str) for idx in row.index):
                    if str(row[feature_name]) != feature_val:
                        return False
        return True

    # format set of rule to dictionary for filter df use
    def pattern_to_dict(self, p):
        """
        Parameters
            p: set, represent rule found by drillUp
        Return
            dict, key:val means col:col_val
        """
        filter_dict = {}
        unknown_col = []
        other_col = []
        for i in sorted(list(p)):
            split_res = i.split(':')
            if split_res[1] == 'unknown':
                unknown_col.append(split_res[0])
            elif split_res[1] == 'other<':
                other_col.append(split_res[0])
            else:
                filter_dict[split_res[0]] = split_res[1]
        return filter_dict, unknown_col, other_col

    # filter unknown result in original data
    def filter_unkown(self, df, unknown_col):
        """
        Parameters
            df: pandas dataframe
            unkown_col: list of col names that are 'unknown' in rules
        Return
            filtered pandas df
        """
        for col in unknown_col:
            logging.info('filter {} with unknown rules'.format(col))
            df = df[df[col].isnull()]
        return df.reset_index(drop=True)

    # filter 'other<' result in original data
    def filter_other(self, df, other_col, map_dict):
        """
        Parameters
            df: pandas dataframe
            other_col: list of col names that are 'other<' in rules
        Return
            filtered pandas df
        """
        for col in other_col:
            logging.info('filter {} with other< rules'.format(col))
            df = df[~df[col].isin(map_dict[col])]
        return df.reset_index(drop=True)

    # format for final output
    def ruleScore(self, test_df):
        """
        Parameters
            rule_df: direct output from drillUp module
            score_df: new data waiting to be scored by rule
            score_date_list: date range list users want to score
            date_col: colname for date in score_df
            label_col: colname of interset, e.g., "gamble_flag"
            label_val: col value to single out point of interest
        Return
            pandas dataframe, with
            [rules, dt, tot_count, poi_count, poi_ratio] as columns
        """
        fill_na = {'rules': 'stophere',
                   'col_has_other_map_no_other': 'stophere'}
        rule_df = self.return_df.fillna(fill_na)

        # format each column in score_df to str in order to filer
        for col in test_df:
            if test_df[col].dtype != object:
                test_df[col] = test_df[col].astype(str)

        per_res = []

        map_dict = {}
        for i in range(rule_df.shape[0]):
            val = rule_df['col_has_other_map_no_other'].values[i]
            if val == 'stophere':
                break
            else:
                # logging.info('new val {}'.format(val))
                # logging.info('map_dict {}'.format(map_dict))
                map_dict.update(eval(val))

        for q, pattern in enumerate(rule_df['rules'].values):
            if pattern == 'stophere':
                break

            logging.info('use pattern {}'.format(pattern))
            rule = ast.literal_eval(pattern)
            rule_dict, unknown_col, other_col = self.pattern_to_dict(rule)
            rule_series = pd.Series(rule_dict)

            filter_df = test_df.loc[(test_df[list(rule_dict)]
                                     == rule_series).all(axis=1)]

            filter_df = self.filter_unkown(filter_df, unknown_col)
            filter_df = self.filter_other(filter_df, other_col, map_dict)

            cnt = filter_df[filter_df[self.label_col] == self.label_val].shape[0]

            per_res.append(
                [pattern, filter_df.shape[0]]
                + [cnt, cnt / filter_df.shape[0]]
            )
        out_df = pd.DataFrame(per_res)
        out_df.columns = ['rules', 'tot_count', self.label_col + '_count',
                          self.label_col + '_ratio']
        out_df["rules"] = out_df["rules"].astype(str)
        return out_df
    def _ensure_dataframe(self, X, columns=None):
        """
        Ensure the input is a pandas DataFrame.
        
        Parameters:
            X (numpy.ndarray, pandas.DataFrame, or pandas.Series): Input data.
            columns (list of str, optional): Column names for the DataFrame if X is a numpy array or Series.
            
        Returns:
            pandas.DataFrame: The input data as a DataFrame.
        """
        if isinstance(X, np.ndarray):
            logging.info("Converting numpy array to DataFrame.")
            return pd.DataFrame(X, columns=columns)  # 如果columns为None，Pandas会自动生成默认列名
        elif isinstance(X, pd.DataFrame):
            return X
        elif isinstance(X, pd.Series):
            logging.info("Converting pandas Series to DataFrame.")
            if columns is not None and isinstance(columns, list) and len(columns) == 1:
                df = pd.DataFrame(X)
                df.columns = columns
                return df
            else:
                columns=['0'] if columns is None else columns
                df = pd.DataFrame(X)
                df.columns = columns
                return df
        else:
            raise TypeError("Input data must be either a numpy array, pandas DataFrame, or pandas Series.")