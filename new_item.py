import pandas as pd
import func_pack
from collections import Counter


def add_type_count(group):
    behavior_type = group.type.astype(int)
    type_cnt = Counter(behavior_type)
    # 1: 浏览 2: 加购 3: 删除
    # 4: 购买 5: 收藏 6: 点击
    group['browse_num'] = type_cnt[1]
    group['addcart_num'] = type_cnt[2]
    group['delcart_num'] = type_cnt[3]
    group['buy_num'] = type_cnt[4]
    group['favor_num'] = type_cnt[5]
    group['click_num'] = type_cnt[6]
    return group[['sku_id', 'browse_num', 'addcart_num',
                  'delcart_num', 'buy_num', 'favor_num',
                  'click_num']]


def get_data_from_action(file_name):
    action_data = pd.read_csv(file_name, encoding='GBK')
    df_stat = action_data.groupby(['sku_id'], as_index=False).apply(add_type_count)
    df_stat = df_stat.drop_duplicates('sku_id')
    return df_stat


def merge_action_data():
    df_data = pd.DataFrame()
    df_data = df_data.append(get_data_from_action(func_pack.file_action02))
    df_data = df_data.append(get_data_from_action(func_pack.file_action03))
    df_data = df_data.append(get_data_from_action(func_pack.file_action04), ignore_index=True)
    df_data = df_data.groupby(['sku_id'], as_index=False).sum()

    # 　构造转化率字段
    df_data['buy_addcart_ratio'] = df_data['buy_num'] / df_data['addcart_num']
    df_data['buy_browse_ratio'] = df_data['buy_num'] / df_data['browse_num']
    df_data['buy_click_ratio'] = df_data['buy_num'] / df_data['click_num']
    df_data['buy_favor_ratio'] = df_data['buy_num'] / df_data['favor_num']
    return df_data


def get_data_from_product():
    df_product = pd.read_csv(func_pack.file_product, encoding='GBK')
    return df_product


def get_data_from_comment():
    df_comment = pd.read_csv(func_pack.file_comment, encoding='GBK')
    df_comment = df_comment.groupby(['sku_id'], as_index=False).sum()
    df_comment['bad_comment_rate'] = df_comment['has_bad_comment'] / df_comment['comment_num']
    return df_comment[['sku_id', 'comment_num', 'has_bad_comment', 'bad_comment_rate']]


def main():
    try:
        product = get_data_from_product()
        product_statistic = merge_action_data()
        product_statistic = pd.merge(product, product_statistic, on=['sku_id'], how='left')
        comment = get_data_from_comment()
        product_statistic = pd.merge(product_statistic, comment, on=['sku_id'], how='left')
        product_statistic.to_csv(func_pack.file_new_product, encoding='GBK', index=False)
        print(product_statistic.head())

    except Exception as e:
        print(str(e))


if __name__ == '__main__':
    main()
