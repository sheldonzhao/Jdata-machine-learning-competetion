import pandas as pd
import func_pack

'''
strategy: linear regression
step:
1. clear users who do not buy anything
2. groupby(['user_id','sku_id']) based on data during 4.11 - 4.15
3. calculate score by linear regression based on the user's behaviour data
submission:
res: 0.010 同样是基于最后一星期的数据，model1是0.022，说明对于用户行为分配的权重还是有问题
res: 0.016 output = output[output['total'] >= 60]
res: 0.022(P=8R) output = output[output['total'] >= 40]
res: 0.041 (total > 2) 0.036 (total > 1.5)
time_weight = {"2016-04-09": 0.5, "2016-04-10": 0.7, "2016-04-11": 0.9, "2016-04-12": 1.1, "2016-04-13": 1.3,
                   "2016-04-14": 1.5, "2016-04-15": 1.7}
type_weight = {1: 0.00773547029882, 2: 0.240584721789, 3: -0.284495324017, 4: -1.62501579407, 5: 0.673595344603,
                   6: 0.00324071075696}
improvement:
1. adjust weights + filter potential user
2. balance P and R within 2:1 from 8:1
'''


def weight(time, type):
    time_weight = {"2016-04-09": 0.5, "2016-04-10": 0.7, "2016-04-11": 0.9, "2016-04-12": 1.1, "2016-04-13": 1.3,
                   "2016-04-14": 1.5, "2016-04-15": 1.7}
    # 1: 浏览 2: 加购 3: 删除
    # 4: 购买 5: 收藏 6: 点击
    type_weight = {1: 0.00773547029882, 2: 0.240584721789, 3: -0.284495324017, 4: -1.62501579407, 5: 0.673595344603,
                   6: 0.00324071075696}
    return time_weight[time.split(' ')[0]] * type_weight[type]


def weight_count(group):
    count = 0
    for index, row in group.iterrows():
        count += weight(row['time'], row['type'])
    group['total'] = count
    return group[['user_id', 'sku_id', 'total']]


def get_from_action_data(fname, start_date, end_date):
    print('start to extract data')
    action_data = pd.read_csv(fname, encoding='GBK')
    action_data = action_data[(action_data['time'] >= start_date) & (action_data['time'] < end_date)]
    return action_data


def merge_action_data(start_date, end_date):
    df_ac = pd.DataFrame()
    df_ac = df_ac.append(get_from_action_data(func_pack.file_action04, start_date, end_date), ignore_index=True)
    # 按ui分组，对每一组进行统计
    print('start to groupby data')
    df_ac = df_ac.groupby(['user_id', 'sku_id'], as_index=False).apply(weight_count)
    df_ac = df_ac.drop_duplicates()
    df_ac.to_csv(func_pack.file_output_temp, index=False)
    return df_ac


def return_max_total_value(group):
    group = group.sort_values('total', ascending=False).head(1)
    return group[['user_id', 'sku_id', 'total']]


def get_highest_score(ui_score):
    ui_score = ui_score.groupby(['user_id'], as_index=False).apply(return_max_total_value)
    return ui_score[['user_id', 'sku_id', 'total']]


if __name__ == '__main__':
    start_date = '2016-04-11'
    end_date = '2016-04-16'
    ui_score = merge_action_data(start_date, end_date)
    ui_score.columns = ['user_id', 'sku_id', 'total']
    output = get_highest_score(ui_score)
    output.to_csv(func_pack.file_output_test, index=False)
