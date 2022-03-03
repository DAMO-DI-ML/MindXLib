#!/usr/bin/env python3

from sklearn.metrics import accuracy_score, f1_score

from mindxlib import *
from mindxlib.ruleset.ruleset_imb import RuleSetImb


def train(X, y):
    clf = RuleSetImb(
        max_num_rules=8,
        factor_g=0.
    )
    clf.fit(X, y)

    return clf

def test(model, X, y):
    y_hat = model.predict(X)
    f1score = f1_score(y, y_hat)
    acc = accuracy_score(y, y_hat)
    rules = model.rules

    return f1score, acc, rules

def main():
    for name in ['tic-tac-toe']:
        df = utils.DatasetLoader(name, basedir='demo/datasets').dataframe

        # Separate target variable
        y = df.pop('label')

        # Binarize the features
        binarizer = utils.FeatureBinarizer(numThresh=9, negations=True, threshStr=True)
        df = binarizer.fit_transform(df)
        df.columns = [' '.join(col).strip() for col in df.columns.values]

        model = train(df, y)
        acc, f1score, rules = test(model, df.to_numpy(), y.to_numpy())

        print("acc = {}, f1score = {}".format(acc, f1score))
        print()
        print("Rules:")
        print(rules)

if __name__ == '__main__':
    main()