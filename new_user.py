import pandas as pd
from collections import Counter
import func_pack


# 功能函数: 对每一个user分组的数据进行统计
def add_type_count(group):
    behavior_type = group.type.astype(int)
    # 用户行为类别
    type_cnt = Counter(behavior_type)
    # 1: 浏览 2: 加购 3: 删除
    # 4: 购买 5: 收藏 6: 点击
    group['browse_num'] = type_cnt[1]
    group['addcart_num'] = type_cnt[2]
    group['delcart_num'] = type_cnt[3]
    group['buy_num'] = type_cnt[4]
    group['favor_num'] = type_cnt[5]
    group['click_num'] = type_cnt[6]
    return group[['user_id', 'browse_num', 'addcart_num',
                  'delcart_num', 'buy_num', 'favor_num',
                  'click_num']]


# 对action数据进行统计
# 根据自己的需求调节chunk_size大小
# 由于用户行为数据量较大,一次性读入可能造成内存错误(Memory Error),因而使用pandas的分块(chunk)读取.
def get_from_action_data(fname, start_date, end_date):
    reader = pd.read_csv(fname, encoding='GBK')
    reader = reader[(reader['time'] >= start_date) & (reader['time'] < end_date)]
    print(reader['time'].unique())
    # 按user_id分组，对每一组进行统计
    df_ac = reader.groupby(['user_id'], as_index=False).apply(add_type_count)
    # 将重复的行丢弃
    df_ac = df_ac.drop_duplicates('user_id')
    return df_ac


# 将各个action数据的统计量进行聚合
def merge_action_data(start_date, end_date):
    df_ac = pd.DataFrame()
    df_ac = df_ac.append(get_from_action_data(func_pack.file_action02, start_date, end_date))
    df_ac = df_ac.append(get_from_action_data(func_pack.file_action03, start_date, end_date))
    df_ac = df_ac.append(get_from_action_data(func_pack.file_action04, start_date, end_date), ignore_index=True)
    # 用户在不同action表中统计量求和
    df_ac = df_ac.groupby(['user_id'], as_index=False).sum()
    # 　构造转化率字段
    df_ac['buy_addcart_ratio'] = df_ac['buy_num'] / df_ac['addcart_num']
    df_ac['buy_browse_ratio'] = df_ac['buy_num'] / df_ac['browse_num']
    df_ac['buy_click_ratio'] = df_ac['buy_num'] / df_ac['click_num']
    df_ac['buy_favor_ratio'] = df_ac['buy_num'] / df_ac['favor_num']
    return df_ac


# 　从JData_User表中抽取需要的字段
def get_from_jdata_user():
    df_usr = pd.read_csv(func_pack.file_user, encoding='GBK')
    df_usr = df_usr[["user_id", "age", "sex", "user_lv_cd"]]
    return df_usr


def main():
    start_date = '2016-01-31'
    end_date = '2016-04-09'
    user = get_from_jdata_user()
    user_behavior = merge_action_data(start_date, end_date)
    # 连接成一张表，类似于SQL的左连接(left join)
    user_behavior = pd.merge(user, user_behavior, on=['user_id'], how='left')
    user_behavior.to_csv(func_pack.file_new_user, encoding='GBK', index=False)
    print(user_behavior.head())


if __name__ == '__main__':
    main()
