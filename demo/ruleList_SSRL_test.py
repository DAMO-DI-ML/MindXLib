import numpy as np
from mindxlib.rulelist.ruleList_SSRL import SSRL
from mindxlib.utils.datautil import DatasetLoader
from mindxlib.utils.features import FeatureBinarizer

para_dict = {
    'tic-tac-toe':{'lambda_1':1,'cc':10,'distorted_step':10}
}
for name, para_v in para_dict.items():
    print('Conduct experments on dataset '+name)
    df = DatasetLoader(name).dataframe
    y = df.pop('label')
    column_list = list(df.columns)
    # Binarize the features
    binarizer = FeatureBinarizer(numThresh=9, negations=True, threshStr=True)
    df = binarizer.fit_transform(df)
    df.columns = [' '.join(col).strip() for col in df.columns.values]
    lambda_1 = para_v['lambda_1']
    cc = para_v['cc']
    distorted_step = para_v['distorted_step']
    model = SSRL(lambda_1=lambda_1,distorted_step=distorted_step,cc=cc,use_multi_pool=True)
    model.fit(df,y,defaultRuleName=0)
    pred_test = model.predict(df)
    acc = np.sum(1.0*(pred_test.values==y.values))/y.shape[0]
    print('The training acc is '+str(acc))
    print('\n')
    print('The learnt rulelist is:')
    model.print_rulelist()