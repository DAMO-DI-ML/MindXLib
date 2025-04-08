import numpy as np
import pandas as pd
from mindxlib import Diver



def test_diver_with_numpy():
    # 创建一个简单的测试数据集
    data = {
        'age': ['young', 'middle-aged', 'old', 'middle-aged', 'old'],
        'income': ['high', 'medium', 'low', 'low', 'medium'],
        'label': [0, 1, 1, 0, 1]  # 0表示正常，1表示异常
    }
    df = pd.DataFrame(data)
    
    X = df.drop('label', axis=1).to_numpy()
    y = df['label'].to_numpy()
    
    # 初始化 Diver 类实例
    driver = Diver(label_col='label', label_val=1, sup_ratio=0.2)
    
    # 拟合模型
    driver.fit(X, y)
    driver.show()
    # 准备测试数据
    test_data = {
        'age': ['young', 'middle-aged', 'old'],
        'income': ['high', 'medium', 'low']
    }
    test_df = pd.DataFrame(test_data).to_numpy()
    
    # 对新数据进行预测
    predictions = driver.predict(test_df)
    print("Predictions:", predictions.tolist())



def test_diver_with_dataframe():
    # 创建一个简单的测试数据集
    data = {
        'age': ['young', 'middle-aged', 'old', 'middle-aged', 'old'],
        'income': ['high', 'medium', 'low', 'low', 'medium'],
        'label': [0, 1, 1, 0, 1]  # 0表示正常，1表示异常
    }
    df = pd.DataFrame(data)
    
    X = df.drop('label', axis=1)
    y = df['label']
    
    # 初始化 Diver 类实例
    driver = Diver(label_col='label', label_val=1, sup_ratio=0.2)
    
    # 拟合模型
    driver.fit(X, y)
    driver.show()
    # 准备测试数据
    test_data = {
        'age': ['young', 'middle-aged', 'old'],
        'income': ['high', 'medium', 'low']
    }
    test_df = pd.DataFrame(test_data)
    
    # 对新数据进行预测
    predictions = driver.predict(test_df)
    print("Predictions:", predictions.tolist())
    


def test_diver_with_dataframe_1():
    ## Examples


    # Create sample data
    # Binary classification example (anomaly detection)
    X = pd.DataFrame({
        'age': [25, 35, 45, 55, 22, 28, 32, 42],
        'income': [30000, 45000, 60000, 75000, 75000, 50000, 65000, 40000],
        'credit_score': [650, 720, 580, 800, 620, 710, 690, 550]
    })
    y = pd.Series([0, 0, 1, 0, 1, 0, 0, 1], name='is_anomaly')  # 1 indicates anomaly

    
    explainer = Diver(
    label_col='is_anomaly',
    label_val=1,  # We're interested in rules for anomalies
    pos_beta=1.5,  # Higher weight on catching anomalies
    overlap_beta_=0.3,  # Moderate penalty for rule overlap
    complexity_cost=0.001,  # Small penalty for longer rules
    dim_list=['age', 'income'],  # Focus on these features
    sup_ratio=0.2  # Minimum 20% support in anomaly class
    )

    explainer.fit(X, y)

def test_diver_from_csv():
    data = pd.read_csv('dataset/tic_tac_toe.csv', header=None)
    y = data.iloc[:,-1]
    print(y.value_counts())
    y = y.map({'negative': 0, 'positive': 1})
    X = data.iloc[:,:-1]
    driver = Diver(label_col='label', label_val=1, pos_beta=2, overlap_beta_=0.2,
                 complexity_cost=0.001,sup_ratio=0.2)

    driver.fit(X, y)
    
    driver.show()
    predictions = driver.predict(X)
    acc = np.sum(1.0*(predictions.values==y.values))/y.shape[0]
    print(f'The training acc is {acc:.2f}')
    '''
    IF 0==b, THEN 1
    ELIF 0==x, THEN 1
    ELIF 1==b, THEN 1
    ELIF 1==x, THEN 1
    ELIF 2==b, THEN 1
    ELIF 2==x, THEN 1
    ELIF 3==b, THEN 1
    ELIF 3==x, THEN 1
    ELIF 4==b, THEN 1
    ELIF 4==x, THEN 1
    ELIF 5==b, THEN 1
    ELIF 5==x, THEN 1
    ELIF 6==b, THEN 1
    ELIF 6==x, THEN 1
    ELIF 7==b, THEN 1
    ELIF 7==x, THEN 1
    ELIF 8==b, THEN 1
    ELIF 8==x, THEN 1
    ELSE 0
    The training acc is 0.65
    '''


if __name__ == "__main__":
    test_diver_with_dataframe_1()