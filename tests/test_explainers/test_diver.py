import numpy as np
import pandas as pd
from mindxlib.explainers.rules.ruleset import Diver



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
    drillup = Diver(label_col='label', label_val=1, sup_ratio=0.2)
    
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



def test_rulelset_with_dataframe():
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
    drillup = Diver(label_col='label', label_val=1, min_dim_val_cnt=1, sup_ratio=0.2)
    
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
    

if __name__ == "__main__":
    test_drillup_with_numpy()