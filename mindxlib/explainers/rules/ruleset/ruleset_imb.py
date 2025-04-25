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
from mindxlib.base.explanation import RuleSetExplanation
from mindxlib.utils.datautil import process_input_data
from mindxlib.utils.features import FeatureBinarizer
from sklearn.base import BaseEstimator
import platform

def get_binary_path():
    system = platform.system().lower()
    machine = platform.machine().lower()
    
    bin_dir = os.path.dirname(os.path.abspath(__file__)) + '/bin/'
    
    if system == 'darwin':  # macOS
        if machine in ('arm64', 'aarch64'):
            return bin_dir + 'new-f1rule-darwin-arm64'
        else:  # x86_64
            return bin_dir + 'new-f1rule-darwin-x86_64'
    elif system == 'linux': # x86_64
            return bin_dir + 'new-f1rule-linux-amd64'
    elif system == 'windows':
            return bin_dir + 'new-f1rule-win-amd64.exe'
    else:
        raise RuntimeError(f"Unsupported platform: {system} on {machine}")

binpath = get_binary_path()


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
        self.label_map = None
        self.reverse_label_map = None

    def _map_labels(self, y):
        """Map labels to 0/1 and store mapping"""
        # Convert y to numpy array first to handle different input types
        if isinstance(y, pd.DataFrame):
            y_values = y.iloc[:,0].values
        elif isinstance(y, pd.Series):
            y_values = y.values
        else:
            y_values = np.array(y)
            
        unique_labels = np.unique(y_values)
        if len(unique_labels) != 2:
            raise ValueError("Only binary classification is supported")
            
        # Create label mapping if not exists
        if self.label_map is None:
            self.label_map = {label: i for i, label in enumerate(unique_labels)}
            self.reverse_label_map = {i: label for label, i in self.label_map.items()}
            
        # Map the values using numpy where for efficiency
        mapped_values = np.array([self.label_map[val] for val in y_values])
        
        # Return as Series to maintain pandas interface
        return pd.Series(mapped_values)

    def fit(self, X, y, default_label=None):
        """Learn a rule set from data
        
        Args:
            X: Input features (DataFrame or ndarray) 
            y: Target labels (required, DataFrame, Series or ndarray)
            default_label: Optional name for default rule (uses most frequent class if None)
            
        Returns:
            RuleSetExplanation object containing the learned rules
        """       
        # Map labels to 0/1
        label_counts = y.value_counts()
        y = self._map_labels(y)

        X, y, feature_columns, label_column = self._process_input_data(X, y, is_fit=True)

        dataset = pd.concat((X, y), axis=1)
        self.feature_columns_ = feature_columns

        # Get default rule name
        
        if default_label is None:
            # convert the name of label_counts to 0/1 via map
            default_label_str = label_counts.idxmax()
            # Handle case where idxmax returns a tuple
            if isinstance(default_label_str, tuple):
                default_label_str = default_label_str[0]
            default_label = self.label_map[default_label_str]
            print(f"Using default rule name: {default_label_str} (most frequent class in data)")
        elif default_label not in list(label_counts.index):
            raise ValueError(f'default_label is not in the data: got {default_label}, expected one of {list(label_counts.index)}')
        else:
            default_label = self.label_map[default_label]
        self.default_label = default_label

        # Store original default label before mapping
        self.original_default_label = self.reverse_label_map[default_label]

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

        # Create explanation with original labels and label mapping
        self.rules = RuleSetExplanation(
            rules=self.ruleset, 
            default_rule=self.original_default_label,
            label_map=self.reverse_label_map
        )

    def predict(self, X: np.ndarray):
        """Make predictions and map back to original labels"""
        X, _, feature_columns, _ = self._process_input_data(X)
        if len(self.ruleset) == 0:
            return np.full(X.shape[0], self.reverse_label_map[0])
            
        predictions = [
            np.prod(X.to_numpy()[..., itemset], axis=-1)
            for itemset in self.itemsets
        ]
        binary_preds = np.greater(np.sum(predictions, axis=0), 0).astype(int)
        
        # Map predictions back to original labels
        return pd.Series(binary_preds).map(self.reverse_label_map)
    
    def _process_input_data(self, X, y=None, is_fit=False):
        """Process input data using utility function"""
        return process_input_data(
            X, 
            y,
            feature_prefix=self.feature_prefix,
            feature_binarizer=self.feature_binarizer,
            is_fit=is_fit
        )
    