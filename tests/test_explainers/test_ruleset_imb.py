import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from mindxlib.explainers.rules.ruleset import RuleSetImb


def test_rulelset_imb_with_numpy():
    # Create numpy array with meaningful feature names
    X = np.array([
        [25, 30000, 12],
        [35, 45000, 14],
        [45, 60000, 16],
        [55, 75000, 18],
        [22, 75000, 18]
    ])
    y = np.array([0, 1, 1, 0, 0])
    
    # Initialize feature binarizer
    # binarizer = FeatureBinarizer(numThresh=3, negations=True, threshStr=True)
    
    # Binarize X while keeping numpy format
    # X_binarized = binarizer.fit_transform(X).values #not working with numpy arrays, as data[c] is not supported
    
    # Initialize and fit explainer with same parameters as pandas test
    explainer = RuleSetImb(
        max_num_rules=3, 
        time_limit=60, 
        verbose=True,
        feature_prefix='feature_',
        binarize_features=True,
        categorical_features=[],
        num_thresh=3,
        negation=True
    )
    
    # Fit the model
    explainer.fit(X, y)
    explainer.show()
    
    X_test = np.array([[25, 30000, 12]])
    a = explainer.predict(X_test)
    print(a)



def test_ruleset_imb_from_csv():
    data = pd.read_csv('dataset/tic_tac_toe.csv', header=None)
    y = data.iloc[:,-1]
    print(y.value_counts())
    y = y.map({'negative': 0, 'positive': 1})
    X = data.iloc[:,:-1]
    explainer = RuleSetImb(
        max_num_rules=15, 
        time_limit=120, 
        verbose=True,
        feature_prefix='feature_',
        binarize_features=True,
        categorical_features=[],
        num_thresh=25,
        negation=True
    )
    explainer.fit(X, y)
    
    explainer.show()
    predictions = explainer.predict(X)
    acc = np.sum(1.0*(predictions.values==y.values))/y.shape[0]
    print(f'The training acc is {acc:.2f}')
    '''
    IF 0!=o AND 4!=o AND 8!=o, THEN 1
    ELIF 2!=o AND 4!=o AND 6!=o, THEN 1
    ELIF 2==x AND 5==x AND 8==x, THEN 1
    ELIF 6==x AND 7==x AND 8==x, THEN 1
    ELIF 0==x AND 3==x AND 6==x, THEN 1
    ELIF 1==x AND 4==x AND 7==x, THEN 1
    ELIF 0!=o AND 1!=o AND 2!=o AND 3!=o AND 4==o AND 5!=x, THEN 1
    ELIF 1!=o AND 3!=o AND 4!=o AND 5!=o AND 7!=o, THEN 1
    ELSE 0
    The training acc is 0.96
    '''



test_ruleset_imb_from_csv()