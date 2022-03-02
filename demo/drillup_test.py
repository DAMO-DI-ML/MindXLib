from mindxlib.ruleset.drillup import DrillUp
from mindxlib.utils.datautil import DatasetLoader

if __name__ == "__main__":
    name = 'tic-tac-toe'
    data = DatasetLoader(name).dataframe
    model = DrillUp(label_col='label', label_val=1, min_pat_len=3, out_num=30)
    model.fit(data)
    print(list(model.return_df.columns))
    print(model.return_df)
    res = model.ruleScore(data)
    print(list(res.columns))