from sklearn.metrics import accuracy_score

from mindxlib.utils.datautil import DatasetLoader
from mindxlib.utils import features
from mindxlib.explainers.rules.ruleset.diver import Diver

if __name__ == "__main__":
    name = 'tic-tac-toe'
    data = DatasetLoader(name).dataframe
    binarizer = features.FeatureBinarizer(numThresh=9, negations=False, threshStr=True)  # negations：是否引入否特征
    onehot_data = binarizer.fit_transform(data)
    onehot_data.columns = [(''.join(col)) for col in onehot_data.columns.values]
    dim_list = list(onehot_data.columns)
    print(dim_list)
    dim_list.remove('label')

    X = onehot_data.drop(['label'], axis=1)
    y_true = onehot_data['label'].astype(int)

    model = Diver(label_col='label', label_val=1,pos_beta=0.8,overlap_beta_=0.0,complexity_cost=0.001)
    model.fit(X, y_true)
    print(model.return_rule)

    y_predict = model.predict(onehot_data[dim_list])
    acc= accuracy_score(y_true, y_predict)
    print(acc)