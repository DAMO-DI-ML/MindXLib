#!/usr/bin/env python3

import argparse
from sklearn.metrics import accuracy_score, f1_score

from mindxlib.utils import DatasetLoader, FeatureBinarizer
from mindxlib.ruleset.ruleset_imb import RuleSetImb

parser = argparse.ArgumentParser()
parser.add_argument('-dataset', dest='dataset', type=str, nargs='+', help='Datset name.')
parser.add_argument('-mxnum', dest='mxnum', type=int, nargs='+', help='Max number of rules.')
parser.add_argument('-lamda', dest='lamda', type=float, nargs='+', help='Lambda added for MM initialization.')
parser.add_argument('-v', dest='verbose', action='store_true', help='Verbose.')
args = parser.parse_args()

def train(X, y):
    clf = RuleSetImb(
        max_num_rules=args.mxnum,
        factor_g=args.lamda,
        warmcache=0, bestsubset=4,
        verbose=args.verbose
    )
    clf.fit(X, y)

    return clf

def test(model, X, y):
    y_hat = model.predict(X)
    f1score = f1_score(y, y_hat)
    acc = accuracy_score(y, y_hat)
    itemsets = model.best_estimator_.itemsets

    return f1score, acc, itemsets

def main():
    for name in args.dataset:
        df = DatasetLoader(name).dataframe

        # Separate target variable
        y = df.pop('label')

        # Binarize the features
        binarizer = FeatureBinarizer(numThresh=9, negations=True, threshStr=True)
        df = binarizer.fit_transform(df)
        df.columns = [' '.join(col).strip() for col in df.columns.values]

        X, y = df.to_numpy(), y.to_numpy()

        model = train(X, y)
        acc, f1score, itemsets = test(model, X, y)

        print("acc = {}, f1score = {}".format(acc, f1score))
        print()
        print("Rules:")
        print(itemsets)

if __name__ == '__main__':
    main()