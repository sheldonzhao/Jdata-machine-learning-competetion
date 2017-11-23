# Jdata-machine-learning-competetion
京东JData算法大赛-高潜用户购买意向预测

## Data Division

train_start_date = '2016-03-10'; train_end_date = '2016-04-11'

test_start_date = '2016-04-11'; test_end_date = '2016-04-16'

prediction_start_date = '2016-03-15'; prediction_end_date = '2016-04-16'

## Feature Engineering
- **User related features: new_user.py**
  - raw features such as age, sex
	- user_browse_num
  - user_addcart_num
  - user_delcart_num
  - user_buy_num
  - user_favor_num
  - user_click_num
  - buy_addcart_ratio
  - buy_browse_ratio
  - buy_click_ratio
  - buy_favor_ratio

- **Item related features: new_item.py**
  - raw features such as brand
  - item_browse_num
  - item_addcart_num
  - item_delcart_num
  - item_buy_num
  - item_favor_num
  - item_click_num
  - buy_addcart_ratio
  - buy_browse_ratio
  - buy_click_ratio
  - buy_favor_ratio
  - comment_num
  - has_bad_comment
  - bad_comment_rate
  
  - **Time related features: gen_feat.py**
  - accumulate_user_feat
  - accumulate_product_feat
  
  - **Statistic related features:
    - sku_hot_rank
    - brank_hot_rank
   
 ## Model and Ensemble
 XGB, LightGBM, Random Forest
 
 Ensemble method: voting
