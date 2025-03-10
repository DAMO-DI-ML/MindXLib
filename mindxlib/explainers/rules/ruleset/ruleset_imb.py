"""
Rule set model for classifying imbalanced data.
"""

import os
import sys
import tempfile
import subprocess as sp

import numpy as np
import pandas as pd
import re
from mindxlib.base.explainer import RuleExplainer
from mindxlib.base.explanation import RuleExplanation
from mindxlib.utils.datautil import process_input_data
from mindxlib.utils.features import FeatureBinarizer
from sklearn.base import BaseEstimator


binpath = os.path.dirname(os.path.abspath(__file__)) + '/bin/f1rule-darwin-aarch64'


class RulesetExplanation(RuleExplanation):
    def show(self):
        """Override show method to print rules in custom format."""
        N = len(self.rules)
        if N > 0:
            # 输出第一个规则
            print(f"IF {self.rules[0]}, THEN 1")
            # 输出其余的规则
            for ii in range(1, N):
                print(f"ELIF {self.rules[ii]}, THEN 1")
            # 输出默认规则
            print(f"ELSE 0")
        else:
            # 如果没有规则，仅输出默认规则
            print(f"IF THEN {self.default_rule}")



class RuleSetImb(RuleExplainer):
    def __init__(
        self, max_num_rules: int=16, time_limit=60, factor_g = 0.0,
        local_search_iter = 0, beta_pos=1.0, beta_neg=1.0, 
        beta_diverse=0.1, beta_complex=0.1,parallelism=0, 
        warmcache=0, bestsubset=0, exactdepth=0, allowrandom=0,
        verbose=False, feature_prefix = 'f',num_thresh=9,negation=True,categorical_features=[],
        binarize_features=True
    ):
        self.max_num_rules = max_num_rules
        self.factor_g = factor_g
        self.local_search_iter = local_search_iter
        self.time_limit = time_limit
        self.beta_pos = beta_pos
        self.beta_neg = beta_neg
        self.beta_diverse = beta_diverse
        self.beta_complex = beta_complex
        self.parallelism = parallelism
        self.warmcache = warmcache
        self.bestsubset = bestsubset
        self.exactdepth = exactdepth
        self.allowrandom = allowrandom
        self.verbose = verbose
        self.feature_prefix = feature_prefix
        self.feature_binarizer = None
        self.num_thresh = num_thresh
        self.negation = negation
        if binarize_features:
            self.feature_binarizer = FeatureBinarizer(
                categorical_features=categorical_features,
                num_thresh=num_thresh,
                negation=negation
            )
    def fit(self, X, y,default_label=None):
        """Learn a rule set from data
        
        Args:
            X: Input features (DataFrame or ndarray) 
            y: Target labels (required, DataFrame, Series or ndarray)
            default_label: Optional name for default rule (uses most frequent class if None)
            
        Returns:
            RuleExplanation object containing the learned rules
        """       

        X, y, feature_columns, label_column = self._process_input_data(X, y, is_fit=True)

        dataset = pd.concat((X, y), axis=1)
        # breakpoint()
        self.feature_columns_ = feature_columns

        # Get default rule name
        label_counts = dataset[label_column].value_counts()
        if default_label is None:
            default_label = label_counts.idxmax()
            print(f"Using default rule name: {default_label} (most frequent class in data)")
        elif default_label not in list(label_counts.index):
            raise ValueError(f'default_label is not in the data: got {default_label}, expected one of {list(label_counts.index)}')
        self.default_label = default_label


        with tempfile.TemporaryFile(mode = "w+") as tmp:
            dataset.to_csv(tmp, index=False)
            tmp.flush()
            tmp.seek(0)
            stderr = sys.stderr.fileno() if self.verbose else sp.DEVNULL
            proc = sp.run([
                binpath,
                '-p', str(self.parallelism),
                '-d', '-',
                '-l', str(label_column),
                '-o', 'h',
                '-k', str(self.max_num_rules),
                '-fac', str(self.factor_g),
                '-iter', str(self.local_search_iter),
                '-t', str(self.time_limit) + 's',
                '-c', str(self.warmcache),
                '-b', str(self.bestsubset),
                '-e', str(self.exactdepth),
                '-r', str(self.allowrandom),
                '-pos', str(self.beta_pos),
                '-neg', str(self.beta_neg),
                '-complex', str(self.beta_complex),
                '-diverse', str(self.beta_diverse),
            ], stdin=tmp, stdout=sp.PIPE, stderr=stderr, check=True)
            lines = proc.stdout.decode("utf-8").splitlines()

        self.itemsets = []
        self.ruleset = []

        for line in lines[:-1]:
            ret = line.split(" <=> ")

            items = re.findall(r'\[(.*?)\]', ret[0])[0]
            if items.strip() != '':
                self.itemsets.append([int(field) for field in items.split(' ')])
            else:
                self.itemsets.append([])

            self.ruleset.append(ret[1])

        self.rules = RulesetExplanation(rules=self.ruleset, default_rule=self.default_label)

    def predict(self, X: np.ndarray):

        X, _, feature_columns, _ = self._process_input_data(X)
        if len(self.ruleset) == 0:
            return np.zeros(X.shape[0], dtype=int)
        predictions = [
            np.prod(X.to_numpy()[..., itemset], axis=-1)
            for itemset in self.itemsets
        ]
        output = pd.DataFrame(np.greater(np.sum(predictions, axis=0), 0).astype(int))
        return output if len(output.shape)==1 else output.iloc[:,0]
    
    def _process_input_data(self, X, y=None, is_fit=False):
        """Process input data using utility function"""
        return process_input_data(
            X, 
            y,
            feature_prefix=self.feature_prefix,
            feature_binarizer=self.feature_binarizer,
            is_fit=is_fit
        )
    