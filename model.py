import pandas as pd
import func_pack

'''
model building strategy
1. clean buy=0 or addcart=0
2. clean buy/browse approaches to 0 (pending)
3. clean people who do not add cart in recent 7 days
4. clean people who have bad buy/addcart
5. analyse users who have behaviour of add cart in 4.9-4.15
6. get the sku_id that user add cart, and user did not del cart and buy it later.
'''


def statistics(group):
    group = group[group['type'] == 2]
    group = group.tail(1)
    return group[['user_id', 'sku_id']]


def analyse_last_week_data():
    df = pd.read_csv(func_pack.file_action04, encoding='GBK')
    # save last week data
    df = df[(df['time'] >= '2016-04-11') & (df['time'] < '2016-04-16')]
    # analyse user behavior
    df = df.groupby('user_id').apply(statistics)
    return df


def cleanUser():
    user = pd.read_csv(func_pack.file_new_user, encoding='GBK')
    user['buy_addcart_ratio'].fillna(-1, inplace=True)
    print(len(user['user_id']))
    # save user who buy_num > 3 and addcart_num != 0
    user = user[(user['buy_num'] >= 3) & (user['addcart_num'] != 0)]
    print(len(user['user_id']))
    # save user who buy/addcart >=0.2
    user = user[(user['buy_addcart_ratio'] >= 0.2)]  # & (user['buy_browse_ratio'] > 0.01)
    print(len(user['user_id']))
    return user


def main():
    # user = cleanUser()
    df = analyse_last_week_data()
    df['user_id'] = df['user_id'].map(int)
    print(len(df['user_id']))
    print(len(df['user_id'].unique()))
    df.to_csv(func_pack.file_output, encoding='GBK', index=False, header=0)


if __name__ == '__main__':
    main()
