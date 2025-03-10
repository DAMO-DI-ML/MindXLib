import numpy as np
import pandas as pd
from mindxlib.explainers.rules.ruleset import DrillUp



'''

The `DrillUp` method processes a list of data, it identifies and removes "bad" columns that are not suitable for further analysis.

Columns are evaluated based on the following criteria to determine if they should be considered "bad":

1. Low Unique Value Count:
   - A column with a unique value count (excluding missing values) less than or equal to 1 (`num_lvls <= 1`) will be identified as a bad column.
   - This indicates that the column either contains almost entirely identical values or has mostly missing data, which provides little useful information for analysis.

2. High Proportion of Unique Values:
   - If the proportion of unique values in a column relative to the total number of rows exceeds 70% (`num_lvls / num > 0.7`), the column is considered a bad column.
   - Such columns often contain a large number of distinct values, such as IDs or timestamps, which may act as noise rather than useful information in certain analytical contexts.

3. High Frequency of a Single Value:
   - Even if a column's unique value count and proportion are within reasonable ranges, if any single value appears more frequently than 95% of the time (`max_ratio > 0.95`), the column is flagged as a bad column.
   - This indicates a highly imbalanced distribution where most data points have the same value, potentially leading to bias or misleading results in many machine learning models.

'''
def test_drillup_with_numpy():
    # 创建一个简单的测试数据集
    data = {
        'age': ['young', 'middle-aged', 'old', 'middle-aged', 'old'],
        'income': ['high', 'medium', 'low', 'low', 'medium'],
        'label': [0, 1, 1, 0, 1]  # 0表示正常，1表示异常
    }
    df = pd.DataFrame(data)
    
    X = df.drop('label', axis=1).to_numpy()
    y = df['label'].to_numpy()
    
    # 初始化 DrillUp 类实例
    drillup = DrillUp(label_col='label', label_val=1, min_dim_val_cnt=1, sup_ratio=0.2)
    
    # 拟合模型
    drillup.fit(X, y)
    drillup.show()
    # 准备测试数据
    test_data = {
        'age': ['young', 'middle-aged', 'old'],
        'income': ['high', 'medium', 'low']
    }
    test_df = pd.DataFrame(test_data).to_numpy()
    
    # 对新数据进行预测
    predictions = drillup.predict(test_df)
    print("Predictions:", predictions.tolist())



def test_drillup_with_dataframe():
    # 创建一个简单的测试数据集
    data = {
        'age': ['young', 'middle-aged', 'old', 'middle-aged', 'old'],
        'income': ['high', 'medium', 'low', 'low', 'medium'],
        'label': [0, 1, 1, 0, 1]  # 0表示正常，1表示异常
    }
    df = pd.DataFrame(data)
    
    X = df.drop('label', axis=1)
    y = df['label']
    
    # 初始化 DrillUp 类实例
    drillup = DrillUp(label_col='label', label_val=1, min_dim_val_cnt=1, sup_ratio=0.2)
    
    # 拟合模型
    drillup.fit(X, y)
    drillup.show()
    # 准备测试数据
    test_data = {
        'age': ['young', 'middle-aged', 'old'],
        'income': ['high', 'medium', 'low']
    }
    test_df = pd.DataFrame(test_data)
    
    # 对新数据进行预测
    predictions = drillup.predict(test_df)
    print("Predictions:", predictions.tolist())


def test_drillup_from_csv():
    data = pd.read_csv('dataset/tic_tac_toe.csv', header=None)
    y = data.iloc[:,-1]
    print(y.value_counts())
    y = y.map({'negative': 0, 'positive': 1})
    X = data.iloc[:,:-1]
    drillup = DrillUp(label_col='label', label_val=1, min_dim_val_cnt=1, sup_ratio=0.2)
    drillup.fit(X, y)
    
    drillup.show()
    predictions = drillup.predict(X)
    acc = np.sum(1.0*(predictions.values==y.values))/y.shape[0]
    print(f'The training acc is {acc:.2f}')
    '''
    IF 2!=o AND 4!=o AND 6!=o, THEN 1
    ELIF 0!=o AND 4!=o AND 8!=o, THEN 1
    ELIF 1!=o AND 2!=o AND 3!=o AND 8!=o, THEN 1
    ELIF 1!=o AND 2==o AND 4!=o AND 5!=o, THEN 1
    ELIF 0!=o AND 2!=o AND 3!=o AND 7!=o, THEN 1
    ELIF 4==o AND 5!=o AND 6!=o AND 7!=o AND 8!=o, THEN 1
    ELIF 0!=o AND 5!=o AND 6!=o AND 7!=o, THEN 1
    ELSE 0
    The training acc is 0.91
    '''

if __name__ == "__main__":
    test_drillup_from_csv()