from datetime import datetime
from datetime import timedelta
import pandas as pd
import math
import xgboost as xgb
import func_pack
from sklearn.model_selection import train_test_split

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


def get_basic_user_feat():
    user = pd.read_csv(func_pack.file_user, encoding='gbk')
    user['age'] = user['age'].map(convert_age)
    age_df = pd.get_dummies(user["age"], prefix="age")
    sex_df = pd.get_dummies(user["sex"], prefix="sex")
    user = pd.concat([user['user_id', 'user_lv_df'], age_df, sex_df], axis=1)
    return user


def get_basic_product_feat():
    product = pd.read_csv(func_pack.file_product, encoding='GBK')
    attr1_df = pd.get_dummies(product["a1"], prefix="a1")
    attr2_df = pd.get_dummies(product["a2"], prefix="a2")
    attr3_df = pd.get_dummies(product["a3"], prefix="a3")
    product = pd.concat([product['sku_id'], attr1_df, attr2_df, attr3_df], axis=1)
    return product


def read_actions(file_name):
    action = pd.read_csv(file_name, encoding='GBK')
    return action


def get_actions(start_date, end_date):
    """
    :param start_date:
    :param end_date:
    :return: actions: pd.Dataframe
    """
    action_1 = read_actions(func_pack.file_action02)
    action_2 = read_actions(func_pack.file_action03)
    action_3 = read_actions(func_pack.file_action04)
    actions = pd.concat([action_1, action_2, action_3])
    actions = actions[(actions['time'] >= start_date) & (actions['time'] < end_date)]
    return actions


def get_action_feat(start_date, end_date):
    actions = get_actions(start_date, end_date)
    actions = actions[['user_id', 'sku_id', 'type']]
    df = pd.get_dummies(actions['type'], prefix='%s-%s-action' % (start_date, end_date))
    actions = pd.concat([actions, df], axis=1)
    actions = actions.groupby(['user_id', 'sku_id'], as_index=False).sum()
    del actions['type']
    return actions


def get_accumulate_action_feat(start_date, end_date):
    actions = get_actions(start_date, end_date)
    df = pd.get_dummies(actions['type'], prefix='action')
    actions = pd.concat([actions, df], axis=1)
    # 近期行为按时间衰减
    actions['weights'] = actions['time'].map(
        lambda x: datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    actions['weights'] = actions['weights'].map(lambda x: math.exp(-x.days))
    actions['action_1'] = actions['action_1'] * actions['weights']
    actions['action_2'] = actions['action_2'] * actions['weights']
    actions['action_3'] = actions['action_3'] * actions['weights']
    actions['action_4'] = actions['action_4'] * actions['weights']
    actions['action_5'] = actions['action_5'] * actions['weights']
    actions['action_6'] = actions['action_6'] * actions['weights']
    del actions['model_id']
    del actions['type']
    del actions['time']
    del actions['weights']
    actions = actions.groupby(['user_id', 'sku_id', 'cate', 'brand'], as_index=False).sum()
    return actions


def get_accumulate_user_feat(start_date, end_date):
    feature = ['user_id', 'user_action_1_ratio', 'user_action_2_ratio', 'user_action_3_ratio',
               'user_action_5_ratio', 'user_action_6_ratio']
    actions = get_actions(start_date, end_date)
    df = pd.get_dummies(actions['type'], prefix='action')
    actions = pd.concat([actions['user_id'], df], axis=1)
    actions = actions.groupby(['user_id'], as_index=False).sum()
    actions['user_action_1_ratio'] = actions['action_4'] / actions['action_1']
    actions['user_action_2_ratio'] = actions['action_4'] / actions['action_2']
    actions['user_action_3_ratio'] = actions['action_4'] / actions['action_3']
    actions['user_action_5_ratio'] = actions['action_4'] / actions['action_5']
    actions['user_action_6_ratio'] = actions['action_4'] / actions['action_6']
    actions = actions[feature]
    return actions


def get_accumulate_product_feat(start_date, end_date):
    feature = ['sku_id', 'product_action_1_ratio', 'product_action_2_ratio', 'product_action_3_ratio',
               'product_action_5_ratio', 'product_action_6_ratio']
    actions = get_actions(start_date, end_date)
    df = pd.get_dummies(actions['type'], prefix='action')
    actions = pd.concat([actions['sku_id'], df], axis=1)
    actions = actions.groupby(['sku_id'], as_index=False).sum()
    actions['product_action_1_ratio'] = actions['action_4'] / actions['action_1']
    actions['product_action_2_ratio'] = actions['action_4'] / actions['action_2']
    actions['product_action_3_ratio'] = actions['action_4'] / actions['action_3']
    actions['product_action_5_ratio'] = actions['action_4'] / actions['action_5']
    actions['product_action_6_ratio'] = actions['action_4'] / actions['action_6']
    actions = actions[feature]
    return actions


def get_labels(start_date, end_date):
    actions = get_actions(start_date, end_date)
    actions = actions[actions['type'] == 4]
    actions = actions.groupby(['user_id', 'sku_id'], as_index=False).sum()
    actions['label'] = 1
    actions = actions[['user_id', 'sku_id', 'label']]
    return actions


def make_test_set(train_start_date, train_end_date):
    start_days = "2016-02-01"
    user = get_basic_user_feat()
    product = get_basic_product_feat()
    user_acc = get_accumulate_user_feat(start_days, train_end_date)
    product_acc = get_accumulate_product_feat(start_days, train_end_date)

    # generate 时间窗口
    actions = None
    for i in (1, 2, 3, 5, 7, 10, 15, 21, 30):
        start_days = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=i)
        start_days = start_days.strftime('%Y-%m-%d')
        if actions is None:
            actions = get_action_feat(start_days, train_end_date)
        else:
            actions = pd.merge(actions, get_action_feat(start_days, train_end_date), how='left',
                               on=['user_id', 'sku_id'])

    actions = pd.merge(actions, user, how='left', on='user_id')
    actions = pd.merge(actions, user_acc, how='left', on='user_id')
    actions = pd.merge(actions, product, how='left', on='sku_id')
    actions = pd.merge(actions, product_acc, how='left', on='sku_id')
    actions = actions.fillna(0)
    users = actions[['user_id', 'sku_id']].copy
    del actions['user_id']
    del actions['sku_id']
    return users, actions


def make_train_set(train_start_date, train_end_date, test_start_date, test_end_date, days=30):
    start_days = "2016-02-01"
    user = get_basic_user_feat()
    product = get_basic_product_feat()
    user_acc = get_accumulate_user_feat(start_days, train_end_date)
    product_acc = get_accumulate_product_feat(start_days, train_end_date)
    labels = get_labels(test_start_date, test_end_date)

    # generate 时间窗口
    actions = None
    for i in (1, 2, 3, 5, 7, 10, 15, 21, 30):
        start_days = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=i)
        start_days = start_days.strftime('%Y-%m-%d')
        if actions is None:
            actions = get_action_feat(start_days, train_end_date)
        else:
            actions = pd.merge(actions, get_action_feat(start_days, train_end_date), how='left',
                               on=['user_id', 'sku_id'])

    actions = pd.merge(actions, user, how='left', on='user_id')
    actions = pd.merge(actions, user_acc, how='left', on='user_id')
    actions = pd.merge(actions, product, how='left', on='sku_id')
    actions = pd.merge(actions, product_acc, how='left', on='sku_id')
    actions = pd.merge(actions, labels, how='left', on=['user_id', 'sku_id'])
    actions = actions.fillna(0)
    users = actions[['user_id', 'sku_id']].copy()
    labels = actions['label'].copy()
    del actions['user_id']
    del actions['sku_id']
    del actions['label']
    return users, actions, labels


def xgboost_make_submission():
    train_start_date = '2016-03-10'
    train_end_date = '2016-04-11'
    test_start_date = '2016-04-11'
    test_end_date = '2016-04-16'
    sub_start_date = '2016-03-15'
    sub_end_date = '2016-04-16'

    user_index, training_data, label = make_train_set(train_start_date, train_end_date, test_start_date, test_end_date)
    X_train, X_test, y_train, y_test = train_test_split(training_data.values, label.values, test_size=0.2,
                                                        random_state=0)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    param = {'learning_rate': 0.1, 'n_estimators': 1000, 'max_depth': 3,
             'min_child_weight': 5, 'gamma': 0, 'subsample': 1.0, 'colsample_bytree': 0.8,
             'scale_pos_weight': 1, 'eta': 0.05, 'silent': 1, 'objective': 'binary:logistic'}
    num_round = 283
    param['nthread'] = 4
    # param['eval_metric'] = "auc"
    # plst = param.items()
    # plst += [('eval_metric', 'logloss')]
    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    bst = xgb.train(param, dtrain, num_round, evallist)
    sub_user_index, sub_trainning_data = make_test_set(sub_start_date, sub_end_date, )
    sub_trainning_data = xgb.DMatrix(sub_trainning_data.values)
    y = bst.predict(sub_trainning_data)
    sub_user_index['label'] = y
    sub_user_index.to_csv(func_pack.file_output_test, index=False, header=0)
    pred = sub_user_index[sub_user_index['label'] >= 0.025]
    sub_user_index.to_csv(func_pack.file_output_test, index=False, header=0)
    pred = pred[['user_id', 'sku_id']]
    pred = pred.groupby('user_id').first().reset_index()
    pred['user_id'] = pred['user_id'].astype(int)
    pred.to_csv(func_pack.file_output, index=False, index_label=False)


if __name__ == '__main__':
    xgboost_make_submission()
