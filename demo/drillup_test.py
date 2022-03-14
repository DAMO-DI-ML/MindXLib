from sklearn.metrics import accuracy_score

from mindxlib.ruleset.drillup import DrillUp
from mindxlib.utils.datautil import DatasetLoader

if __name__ == "__main__":
    name = 'tic-tac-toe'
    data = DatasetLoader(name).dataframe
    model = DrillUp(label_col='label', label_val=1, min_pat_len=3, out_num=10)
    model.fit(data)
    print(model.return_df)
    print(model.output_rule)
    res = model.ruleScore(data)
    print(list(res.columns))

    X = data.drop(['label'],axis=1)
    y_true = data['label'].astype(int)
    y_predict = model.predict(X)
    acc = accuracy_score(y_true, y_predict)
    print(acc)