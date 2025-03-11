import numpy as np
from mindxlib.explainers.rules.rulelist import SSRL
import pandas as pd
from mindxlib.utils.features import FeatureBinarizer


data = pd.read_csv('dataset/tic_tac_toe.csv', header=None)
y = data.iloc[:,-1]
X = data.iloc[:,:-1]
explainer = SSRL(cc=10, lambda_1=1, distorted_step=10, categorical_features=X.columns.tolist())
explainer.fit(X, y)
explainer.show()
    