# load data to mysql

import pymysql
import pandas as pd

# sku_id	attr1	attr2	attr3	cate	brand
try:
    conn = pymysql.connect(host='localhost', user='root', passwd='', port=3306)
    cur = conn.cursor()
    conn.select_db('JDdata')
    file_product = '/Users/XaviZhao/Desktop/kaggle/JD/JData_Product/JData_Product.csv'
    data = pd.read_csv(file_product, encoding='GBK')
    sku_id = [str(i) for i in data['sku_id']]
    attr1 = [str(i) for i in data['attr1']]
    attr2 = [str(i) for i in data['attr2']]
    attr3 = [str(i) for i in data['attr3']]
    cate = [str(i) for i in data['cate']]
    brand = [str(i) for i in data['brand']]
    # combine values in the every row to tuple
    value = []
    for i in range(len(data)):
        value.append((sku_id[i], attr1[i], attr2[i], attr3[i], cate[i], brand[i]))
    # insert data to mysql
    cur.executemany('insert into Product values(%s,%s,%s,%s,%s,%s)', value)
    conn.commit()
    cur.close()
    conn.close()

except Exception as e:
    print(str(e))
