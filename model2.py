import func_pack
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier

'''
strategy:
1. classify the people who will buy during 4.16-4.20
2. choose the item they are most likely to buy
step:
1. generate user features first
2. fit the people who will buy during 4.09-4.15 based on the data during 2.01-4.08
3. predict the people who will buy during 4.16-4.20 based on the data during 2.01-4.15
method: random forest
shortcoming: lose key dim - time
training: start_date='2016-01-31' end_date='2016-04-09'
test: start_date='2016-04-09' end_date='2016-04-16'
# result: 0.003 决策树效果非常差，因为没有时间变量，必须用线性回归筛选购买人群，越近时间段的行为权重越大
'''


def read_user_data():
    user_df = pd.read_csv('/Users/XaviZhao/Desktop/kaggle/JData/new_user2016-01-31:2016-04-09.csv', encoding='GBK')
    print(user_df.shape)
    return user_df


def convert_age(age_str):
    if age_str == u'-1':
        return 0
    elif age_str == u'15岁以下':
        return 1
    elif age_str == u'16-25岁':
        return 2
    elif age_str == u'26-35岁':
        return 3
    elif age_str == u'36-45岁':
        return 4
    elif age_str == u'46-55岁':
        return 5
    elif age_str == u'56岁以上':
        return 6
    else:
        return -1


def preprocess(user):
    # discretize user age
    user['age'] = user['age'].map(convert_age)
    # delete the users who have no info
    user = user[user['browse_num'].notnull()]
    # fill NAN
    user.loc[user['sex'].isnull(), 'sex'] = 2
    user = user.fillna(0)
    # replace inf to 0
    user = user.replace(np.inf, 0)
    # scaling
    '''
    scaler = preprocessing.StandardScaler()
    age_scale_param = scaler.fit(data_train['Age'])
    data_train['Age_scaled'] = scaler.fit_transform(data_train['Age'], age_scale_param)
    '''
    # dummy age, sex
    age_df = pd.get_dummies(user['age'], prefix='age')
    sex_df = pd.get_dummies(user['sex'], prefix='sex')
    user.pop('age')
    user.pop('sex')
    user = pd.concat([user, age_df, sex_df], axis=1)
    return user


def set_label():
    data = pd.read_csv(func_pack.file_action04, encoding='GBK')
    data = data[(data['time'] >= '2016-04-09') & (data['time'] < '2016-04-16')]
    user_id = data[data['type'] == 4]['user_id'].unique()
    return user_id


def model_random_forest():
    user = read_user_data()
    X_train = preprocess(user)
    X_train['label'] = 0
    user_id = set_label()
    # TODO: balance postive and negative sample now:47:1000
    X_train.loc[X_train['user_id'].isin(user_id), 'label'] = 1
    y_label = X_train.pop('label')
    X_train.pop('user_id')
    # TODO: max_features=params
    clf = RandomForestClassifier(n_estimators=1000, max_features=0.3)
    clf.fit(X_train, y_label)
    return clf


def preprocess_test():
    user_df = pd.read_csv(func_pack.file_new_user, encoding='GBK')
    user_df = preprocess(user_df)
    return user_df

def output():
    user = pd.read_csv(func_pack.file_output_temp)
    print(user.shape)
    user.columns = ['user_id', 'label']
    action = pd.read_csv(func_pack.file_action04, encoding='GBK')
    action = action[action['type'] == 2]
    action = action[(action['time'] >= '2016-04-09') & (action['time'] < '2016-04-16')]
    print(len(action['user_id']))
    merge_data = pd.merge(user, action, how='left', on='user_id')
    merge_data = merge_data.loc[merge_data['sku_id'].notnull()]
    merge_data = merge_data.groupby(['user_id'], as_index=False).last()
    merge_data['sku_id'] = merge_data['sku_id'].astype(int)
    merge_data = merge_data.ix[:, ['user_id', 'sku_id']]
    print(merge_data.shape)
    merge_data.to_csv(func_pack.file_output, index=False, header=0)


if __name__ == '__main__':
    clf = model_random_forest()
    test_user_df = preprocess_test()
    user_id = test_user_df.pop('user_id')
    res = clf.predict(test_user_df)
    res = pd.Series(res)
    res = pd.concat([user_id, res], axis=1)
    res.columns = ['user_id', 'label']
    res = res[res['label'] == 1]
    res['user_id'] = res['user_id'].astype(int)
    res['label'] = res['label'].astype(int)
    res.to_csv(func_pack.file_output_temp, index=False, header=0)
    print(res.shape)
    output()
