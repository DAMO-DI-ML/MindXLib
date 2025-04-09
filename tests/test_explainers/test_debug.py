import pandas as pd
import numpy as np
from mindxlib import SSRL
from mindxlib.data import tic_tac_toe

# Load tic-tac-toe dataset
X, y = tic_tac_toe()

# 初始化并训练SSRL
explainer = SSRL(cc=10, lambda_1=1, distorted_step=10, 
                categorical_features=X.columns.tolist())
explainer.fit(X, y)

# 展示学习到的规则
explainer.show()

# 进行预测
predictions = explainer.predict(X)
acc = np.sum(predictions.values == y.values) / y.shape[0]
print(f'训练准确率: {acc:.2f}')