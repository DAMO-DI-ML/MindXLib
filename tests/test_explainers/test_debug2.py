from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
credit_approval = fetch_ucirepo(id=27) 
  
# data (as pandas dataframes) 
X = credit_approval.data.features 
y = credit_approval.data.targets 
# 给x的dummy特征做one-hot处理
import pandas as pd
X = pd.get_dummies(X)
# transfer y to 0 and 1
y = [1*(x=='+') for x in y.values.flatten()]
  
from sklearn.model_selection import train_test_split
import numpy as np
X.dropna(axis=1, inplace=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

 # import XGBClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

model = XGBClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f'训练准确率: {acc:.2f}')



