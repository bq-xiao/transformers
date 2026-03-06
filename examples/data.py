# 读取CSV文件
from datasets import load_dataset

csv_dataset = load_dataset(path='csv',
                           data_files='../raw_data/sku_data.csv',
                           split='train'
                           )
# data = pd.read_csv('../raw_data/sku_data.csv')
for l in csv_dataset:
    str = "{}在{}为公司{}下的门店{}，收货地址是{}，下单{}多少{}".format(
        l['order_creator_name'],
        l['delivery_date'],
        l['company_short_name'],
        l['delivery_short_name'],
        l['receive_address'],
        l['sku_name'],
        l['unit_name'],
    )
    str1 = "{}{}".format(
        l['order_quantity'],
        l['unit_name']
    )
    print("{}@@{}".format(str, str1))
