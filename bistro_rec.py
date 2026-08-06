import pandas as pd

#建立餐酒館的數據菜單

bistro_menu = {
    "item_id" : ["M01" , "M02" , "W01" , "W02"] ,
    "item_name" :["舒肥紅酒燉牛肉" , "紹興油泡蝦" , "馬爾貝克紅酒" , "夏多內白酒"] ,
    "type" : ["主食", "副食" , "酒水" , "酒水"] ,
    "pairing_tag" :["Red_Meat" , "Side" , "Red_Meat" , "Seafood"] ,
    "profit_margin" : [0.4 , 0.7 , 0.6 , 0.6]
}

df_bistro_menu = pd.DataFrame(bistro_menu)

print("你得第一個餐酒館 POS 菜單資料庫建立成功")
print(df_bistro_menu)