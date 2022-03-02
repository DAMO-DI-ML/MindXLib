from mindxlib.pre_mining.wkmodes_rule import extract_wkmodes
from mindxlib.utils.datautil import DatasetLoader

if __name__ == "__main__":
    name = 'tic-tac-toe'
    data = DatasetLoader(name).dataframe
    print(list(data.columns))
    print(data.head(5))
    res = extract_wkmodes(data, 'label', 1)
    print(res)