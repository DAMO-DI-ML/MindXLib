from mindxlib.pre_mining.extract_rule import extract_fpgrowth, extract_rf
from mindxlib.utils import features
from mindxlib.utils.datautil import DatasetLoader

if __name__ == "__main__":
    name = 'tic-tac-toe'
    data = DatasetLoader(name).dataframe
    print(list(data.columns))
    y = data.pop('label')
    X = data.copy()
    binarizer = features.FeatureBinarizer(numThresh=9, negations=False, threshStr=True)  # negations：是否引入否特征
    onehot_X = binarizer.fit_transform(data)
    onehot_X.columns = [(''.join(col)) for col in onehot_X.columns.values]

    res = extract_fpgrowth(onehot_X, maxcardinality=3)
    print(res)

    res2 = extract_rf(onehot_X, y, 1)
    print(res2)