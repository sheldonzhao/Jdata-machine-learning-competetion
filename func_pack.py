import pandas as pd

file_action02 = '/Users/XaviZhao/Desktop/kaggle/JData/JData_Action_201602.csv'
file_action03 = '/Users/XaviZhao/Desktop/kaggle/JData/JData_Action_201603.csv'
file_action04 = '/Users/XaviZhao/Desktop/kaggle/JData/JData_Action_201604.csv'
file_user = '/Users/XaviZhao/Desktop/kaggle/JData/JData_User.csv'
file_product = '/Users/XaviZhao/Desktop/kaggle/JData/JData_Product.csv'
file_comment = '/Users/XaviZhao/Desktop/kaggle/JData/JData_Comment.csv'
file_new_user = '/Users/XaviZhao/Desktop/kaggle/JData/new_user.csv'
file_new_product = '/Users/XaviZhao/Desktop/kaggle/JData/new_product.csv'
file_output = '/Users/XaviZhao/Desktop/kaggle/JData/output.csv'
file_output_temp = '/Users/XaviZhao/Desktop/kaggle/JData/output_temp.csv'
file_output_test = '/Users/XaviZhao/Desktop/kaggle/JData/output_test.csv'


def printDataFrame(df):
    for index, row in df.iterrows():
        print(row)


def read_data_from_action(file_name):
    data = pd.read_csv(file_name, encoding='GBK')
    return data
