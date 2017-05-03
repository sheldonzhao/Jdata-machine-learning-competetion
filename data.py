import pandas as pd
import matplotlib.pyplot as plt
import func_pack


# Feb, Mar, Apr purchase table
def filter_data(data):
    data = data[data['type'] == 4]
    return data[['user_id', 'sku_id', 'time']]


def df_action(data):
    action = pd.DataFrame()
    action = action.append(filter_data(data))
    action['time'] = pd.to_datetime(action['time']).apply(lambda x: x.day)
    return action


def df_user(action):
    df_user = action.groupby(['time'])['user_id'].nunique()
    df_user = df_user.to_frame().reset_index()
    df_user.columns = ['day', 'user_num']
    return df_user


def df_item(action):
    df_item = action.groupby('time')['sku_id'].nunique()
    df_item = df_item.to_frame().reset_index()
    df_item.columns = ['day', 'item_num']
    return df_item


def df_ui(action):
    df_ui = action.groupby('time', as_index=False).size()
    df_ui = df_ui.to_frame().reset_index()
    df_ui.columns = ['day', 'user_item_num']
    return df_ui


def merge_data():
    data02 = func_pack.read_data_from_action(func_pack.file_action02)
    data03 = func_pack.read_data_from_action(func_pack.file_action03)
    data04 = func_pack.read_data_from_action(func_pack.file_action04)
    df_action02 = df_action(data02)
    df_action03 = df_action(data03)
    df_action04 = df_action(data04)

    # draw
    bar_width = 0.2
    opacity = 0.4
    # 天数
    # 设置图片大小
    plt.figure(figsize=(15, 10))
    plt.subplot2grid((3, 1), (0, 0))
    plt.title('Feb Purchase')
    plt.bar(df_user(df_action02)['day'], df_user(df_action02)['user_num'], bar_width,
            alpha=opacity, color='c', label='user')
    plt.bar(df_item(df_action02)['day'] + bar_width, df_item(df_action02)['item_num'],
            bar_width, alpha=opacity, color='g', label='item')
    plt.bar(df_ui(df_action02)['day'] + bar_width * 2, df_ui(df_action02)['user_item_num'],
            bar_width, alpha=opacity, color='m', label='user_item')

    plt.subplot2grid((3, 1), (1, 0))
    plt.title('March Purchase')
    plt.bar(df_user(df_action03)['day'], df_user(df_action03)['user_num'], bar_width,
            alpha=opacity, color='c', label='user')
    plt.bar(df_item(df_action03)['day'] + bar_width, df_item(df_action03)['item_num'],
            bar_width, alpha=opacity, color='g', label='item')
    plt.bar(df_ui(df_action03)['day'] + bar_width * 2, df_ui(df_action03)['user_item_num'],
            bar_width, alpha=opacity, color='m', label='user_item')

    plt.subplot2grid((3, 1), (2, 0))
    plt.title('April Purchase')
    plt.bar(df_user(df_action04)['day'], df_user(df_action04)['user_num'], bar_width,
            alpha=opacity, color='c', label='user')
    plt.bar(df_item(df_action04)['day'] + bar_width, df_item(df_action04)['item_num'],
            bar_width, alpha=opacity, color='g', label='item')
    plt.bar(df_ui(df_action04)['day'] + bar_width * 2, df_ui(df_action04)['user_item_num'],
            bar_width, alpha=opacity, color='m', label='user_item')

    plt.xlabel('day')
    plt.ylabel('number')
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    merge_data()


if __name__ == '__main__':
    main()
