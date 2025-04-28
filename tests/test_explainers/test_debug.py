from mindxlib import RuleSetImb
from mindxlib.data import tic_tac_toe
import numpy as np
from sklearn.model_selection import train_test_split

# 加载井字棋数据集
X, y = tic_tac_toe()

# 查看类别分布
print(y.value_counts())
# 1    626  # 正类(获胜)样本数
# 0    332  # 负类(失败)样本数

# 数据集分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练RuleSetImb模型
model = RuleSetImb(
    num_thresh=1,
    negation=True,
    max_num_rules=20,
    beta_diverse=0.01
)
model.fit(X_train, y_train, default_label='negative')

# 评估模型效果
predictions = model.predict(X_test)
acc = np.sum(1.0*(predictions.values==y_test.values))/y_test.shape[0]
print(f'准确率: {acc:.2f}')  # 输出: 准确率: 1.00
# 展示规则列表
model.show()