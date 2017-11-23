import pandas as pd
import func_pack
from gen_feat import get_actions


def read_file(file_name):
    data = pd.read_csv(file_name, encoding='GBK')
    return data


def clear_user():
    user = read_file(func_pack.file_new_user)
    user = user[user['buy_num'] != 0]  # almost 30% of all users
    print('the number of users who buy the product %d' % len(user['user_id']))
    return user['user_id']


# stat high potential user
def stat_days(group):
    '''
    what is high potential user?
    '''
    if group[group['type'] == 4].empty or group[group['type'] == 2].empty:
        group['high_potential'] = 0
        return group[['user_id', 'sku_id', 'high_potential']]
    else:
        group['time'] = pd.to_datetime(group['time']).apply(lambda x: x.day)
        mean_4 = group[group['type'] == 4]['time'].mean()
        mean_2 = group[group['type'] == 2]['time'].mean()
        if 0 < mean_4 - mean_2 <= 3:
            group['high_potential'] = 1
        else:
            group['high_potential'] = 0
        return group[['user_id', 'sku_id', 'high_potential']]


def stat_buy_freq(group):
    if group[group['type'] == 4].empty:
        group['high_potential'] = 0
    else:
        group = group[group['type'] == 4]
        group['week_1'] = 0
        group['week_2'] = 0
        group['week_3'] = 0
        group.loc[(group['time'] >= '2016-03-26') & (group['time'] < '2016-04-02'), 'week_1'] = 1
        group.loc[(group['time'] >= '2016-04-02') & (group['time'] < '2016-04-09'), 'week_2'] = 1
        group.loc[(group['time'] >= '2016-04-09') & (group['time'] < '2016-04-16'), 'week_3'] = 1
        count = 0
        if sum(group['week_1']):
            count += 1
        if sum(group['week_2']):
            count += 1
        if sum(group['week_3']):
            count += 1
        if count >= 2:
            group['high_potential'] = 1
        else:
            group['high_potential'] = 0
    return group[['user_id', 'sku_id', 'high_potential']]


def high_petential_user():
    user = clear_user()
    # start_date = '2016-03-26'
    start_date = '2016-03-26'
    end_date = '2016-04-16'
    actions = get_actions(start_date, end_date)
    user_action = pd.merge(user.to_frame(), actions, how='left', on='user_id')
    user_action = user_action.fillna(0)
    user_action = user_action[user_action['type'] != 0]
    user_action = user_action.groupby('user_id', as_index=False).apply(stat_buy_freq)
    user_action = user_action[user_action['high_potential'] == 1]
    return user_action


def h_user_sku():
    data = pd.read_csv(func_pack.file_output_test)
    user = pd.read_csv(func_pack.file_output_h_user)
    data.columns = ['user_id', 'sku_id', 'label']
    user_sku = pd.merge(user, data, how='left', on='user_id')
    user_sku = user_sku.groupby('user_id', as_index=False).apply(func_pack.select_highest_score)
    print(len(user_sku['user_id']))  # 72
    user_sku = user_sku.fillna(0)
    user_sku = user_sku[user_sku['label'] != 0]
    user_sku = func_pack.format_data(user_sku)
    user_sku = user_sku[['user_id', 'sku_id']]
    res = read_file(func_pack.file_output_0085)
    res = pd.concat([res, user_sku])
    res.to_csv(func_pack.file_output)
    func_pack.file_validation(func_pack.file_output)
    return user_sku


def duplicate_buy():
    # useless feature
    start_date_3 = '2016-03-16'
    end_date_3 = '2016-03-21'
    start_date_4 = '2016-02-16'
    end_date_4 = '2016-02-21'
    action_3 = get_actions(start_date_3, end_date_3)
    action_2 = get_actions(start_date_4, end_date_4)
    action_3 = action_3[action_3['type'] == 4]
    action_3['label'] = 0
    action_2 = action_2[action_2['type'] == 4]
    action_2['label'] = 0
    user_sku_3 = action_3.groupby(['user_id', 'sku_id'], as_index=False).sum()
    user_sku_3 = user_sku_3.drop_duplicates()
    user_sku_3 = user_sku_3[['user_id', 'sku_id']]
    user_sku_2 = action_2.groupby(['user_id', 'sku_id'], as_index=False).sum()
    user_sku_2 = user_sku_2.drop_duplicates()
    user_sku_2 = user_sku_2[['user_id', 'sku_id']]
    total = pd.concat([user_sku_2, user_sku_3])
    total['label'] = 1
    total = total.groupby(['user_id', 'sku_id'], as_index=False).sum()
    total = total[total['label'] >= 2]
    total.to_csv(func_pack.file_output_temp, index=False)
    print(len(total['user_id']))


if __name__ == '__main__':
    h_user_sku()
