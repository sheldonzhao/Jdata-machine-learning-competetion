import pandas as pd
import func_pack
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np

'''
strategy: linear regression + cv
step:
1. groupby([user_id','sku_id']) based on data during 4.4 - 4.8
2. fit the buy data during 4.9-4.13
3. use the logistic and random forest model to predict 4.16-4.20 based on the data during 4.11-4.15
submission:
Fail. For logistic Regression model, it only predicts 26 "1", and RF predicts 300+ "1".
improvement:
try xgboost and GBDT
'''


def type_count(group):
    len_1 = len(group[group['type'] == 1])
    len_2 = len(group[group['type'] == 2])
    len_3 = len(group[group['type'] == 3])
    len_4 = len(group[group['type'] == 4])
    len_5 = len(group[group['type'] == 5])
    len_6 = len(group[group['type'] == 6])
    group['browse_num'] = len_1
    group['add_cart_num'] = len_2
    group['del_cart_num'] = len_3
    group['buy_num'] = len_4
    group['favorite_num'] = len_5
    group['click_num'] = len_6
    return group[
        ['user_id', 'sku_id', 'browse_num', 'add_cart_num', 'del_cart_num', 'buy_num', 'favorite_num', 'click_num']]


def get_from_action_data(fname, start_date, end_date):
    print('start to extract data from %s to %s' % (start_date, end_date))
    print(datetime.now())
    action_data = pd.read_csv(fname, encoding='GBK')
    action_data = action_data[(action_data['time'] >= start_date) & (action_data['time'] < end_date)]
    return action_data


def merge_action_data(start_date, end_date):
    df_ac = pd.DataFrame()
    df_ac = df_ac.append(get_from_action_data(func_pack.file_action04, start_date, end_date), ignore_index=True)
    print('start to groupby data')
    print(datetime.now())
    df_ac = df_ac.groupby(['user_id', 'sku_id'], as_index=False).apply(type_count)
    df_ac = df_ac.drop_duplicates()
    return df_ac


def statistic(group):
    if group[group['type'] == 4].empty:
        group['label'] = 0
    else:
        group['label'] = 1
    return group[['user_id', 'sku_id', 'label']]


def get_label(start_test_date, end_test_date):
    action_data = get_from_action_data(func_pack.file_action04, start_test_date, end_test_date)
    action_data = action_data.groupby(['user_id', 'sku_id'], as_index=False).apply(statistic)
    action_data = action_data.drop_duplicates()
    action_data['user_id'] = action_data['user_id'].astype(int)
    return action_data


def merge_train_test(training_set, y_label):
    table = pd.merge(y_label, training_set, how='left', on=['user_id', 'sku_id'])
    table = table.fillna(0)
    table = table[table['browse_num'] != 0]
    return table


def analysis(df_test, coef):
    param = pd.DataFrame({"columns": list(df_test.columns), "coef": list(clf.coef_.T)})
    # coef 为正，结果正相关， 为负，负相关
    print('===== coef =====')
    print(param)


if __name__ == '__main__':
    print(datetime.now())
    # if buy during 4.9 - 4.13, label = 1, else: label = 0
    y_label = get_label('2016-04-09', '2016-04-14')
    print(y_label.shape)
    # create features based on the action data during 4.4 - 4.8
    action_stat = merge_action_data('2016-04-04', '2016-04-09')
    print(action_stat.shape)
    # merge label and features
    training_set = merge_train_test(action_stat, y_label)
    training_set = training_set.fillna(0)
    print(training_set.shape)
    # train model
    X = training_set[['browse_num', 'add_cart_num', 'del_cart_num', 'buy_num', 'favorite_num', 'click_num']]
    y = training_set['label']
    # clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    clf = clf = RandomForestClassifier(n_estimators=1000, max_features=0.3)
    clf.fit(X, y)
    # analysis(X, clf)
    # create test_set during 4.11 - 4.15
    test_stat = merge_action_data('2016-04-11', '2016-04-16')
    print(test_stat.shape)
    test_set = test_stat[['browse_num', 'add_cart_num', 'del_cart_num', 'buy_num', 'favorite_num', 'click_num']]
    predictions = clf.predict(test_set)
    submission = pd.DataFrame({'user_id': test_stat['user_id'], 'sku_id': test_stat['sku_id'], 'label': predictions})
    submission.to_csv(func_pack.file_output_temp, index=False, header=0)
    print(datetime.now())
    # cross validation
    max_features = [0.1, 0.3, 0.5, 0.7, 0.9]
    test_scores = []
    for max_feat in max_features:
        clf = RandomForestClassifier(n_estimators=1000, max_features=max_feat)
        test_score = np.sqrt(-cross_val_score(clf, X, y, cv=5, scoring='neg_mean_squared_error'))
        test_scores.append(np.mean(test_score))
    plt.plot(max_features, test_scores)
    plt.show()
