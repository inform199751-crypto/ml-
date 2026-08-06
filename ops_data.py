"""
連鎖餐飲叫貨與成本控管 —— 共用資料層

對應《連鎖餐飲ML叫貨成本控管_技術規格書》第三節「資料規格」。

這一層做兩件事:
  1. 把 河岸餐酒館_營運分析.xlsx 讀進來,欄位名稱轉成規格書 3.1 的標準命名
     (品項 -> sku_id、數量 -> qty_sold ...),讓後面三個模組不用管資料從哪來
  2. 供應商主檔缺失時,依規格書 4.2「例外處理」的規定,
     以**品類預設值**代入並標記 needs_data_fill=True,不阻擋叫貨建議產生

為什麼要獨立成一層
------------------
規格書 3.2 的資料倉儲是星型架構(Fact_Sales / Fact_Inventory / Dim_Supplier ...),
現在資料還在一個 Excel 檔裡。把「讀資料」和「算邏輯」分開,
之後 ETL 建好、資料改從資料倉儲來,只要改這個檔,三個模組一行都不用動。

這正是導入專案該有的做法:**先把介面切乾淨,再換底層。**
"""

import os
import sys

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
XLSX = os.path.join(BASE_DIR, "河岸餐酒館_營運分析.xlsx")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

STORE_ID = "S01"          # 單店資料;規格書的模型是多分店,欄位先留著
ANALYSIS_MONTH = "2026-07"


# ------------------------------------------------------------------
# 供應商品類預設值(規格書 3.1.5 Dim_Supplier / 4.2 例外處理)
# ------------------------------------------------------------------
# 真實供應商主檔還沒建(工作任務清單 P1-03),所以先用品類預設值。
# 每一個數字都標了理由,之後採購補完真實資料就換掉。
#
#   lead_time_days_avg/std  前置天數的平均與標準差
#                           -> std 會進安全庫存公式,交期越不穩要備越多
#   moq                     最小訂購量
#   pack_size               訂購包裝單位(叫貨量要進位到它的倍數)
SUPPLIER_DEFAULTS = {
    "酒水": dict(lead_time_days_avg=5.0, lead_time_days_std=1.2,
                 moq=12, pack_size=12, payment_term_days=45),
    # 酒商配送最慢、但整箱進貨、又不會壞 -> 前置期長不可怕,可以囤
    "前菜": dict(lead_time_days_avg=2.0, lead_time_days_std=0.5,
                 moq=20, pack_size=10, payment_term_days=30),
    "主餐": dict(lead_time_days_avg=2.0, lead_time_days_std=0.6,
                 moq=10, pack_size=5, payment_term_days=30),
    # 主餐是肉品海鮮,交期變異比蔬菜大(市場行情、船期)
    "甜點": dict(lead_time_days_avg=1.0, lead_time_days_std=0.3,
                 moq=12, pack_size=6, payment_term_days=30),
}
CATEGORY_FALLBACK = dict(lead_time_days_avg=2.0, lead_time_days_std=0.5,
                         moq=10, pack_size=1, payment_term_days=30)

# 保存期限(天)—— 叫貨量的硬上限,超過的部分注定變廚餘
SHELF_LIFE_DAYS = {
    "酒水": 365,    # 實務上視為不會壞
    "前菜": 5,      # 冷凍/半成品
    "主餐": 3,      # 肉品海鮮,最不能囤
    "甜點": 3,      # 手作,冷藏保存有限
}

# 食材成本率的業界標竿(規格書 4.4 / 企劃書效益基準)
FOOD_COST_BENCHMARK = (0.28, 0.35)


# ------------------------------------------------------------------
# 讀取
# ------------------------------------------------------------------
def load_sales_daily(path=XLSX):
    """
    銷售明細 -> Fact_Sales(規格書 3.1.1)

    回傳欄位:store_id, sku_id, category, date, qty_sold, unit_price, amount
    """
    df = pd.read_excel(path, sheet_name="銷售明細")
    df = df.rename(columns={
        "日期": "date", "品項": "sku_id", "分類": "category",
        "數量": "qty_sold", "單價(元)": "unit_price", "小計(元)": "amount",
    })
    df["date"] = pd.to_datetime(df["date"])
    df["store_id"] = STORE_ID
    df["weekday"] = df["date"].dt.weekday
    return df[["store_id", "sku_id", "category", "date", "weekday",
               "qty_sold", "unit_price", "amount"]]


def load_inventory(path=XLSX):
    """
    庫存盤點 -> Fact_Inventory(規格書 3.1.6)

    ⚠️ 這裡刻意**不沿用原表的「差異率」欄位**。
    原表的差異率 = 差異 ÷ 理論庫存,分母是期末剩餘量。
    期末剩餘量小的品項(例如主廚牛排只剩 7 份)會被算出誇張的百分比,
    造成假警報 —— 詳見 stocktake.py 的說明。
    正確的差異率定義在規格書 4.5,由 stocktake.py 重新計算。
    """
    df = pd.read_excel(path, sheet_name="庫存盤點")
    df = df.rename(columns={
        "品項": "sku_id", "分類": "category", "單位": "uom",
        "期初庫存(份)": "opening_stock",
        "本月叫貨量(份)": "received_qty",
        "本月銷售量(份)": "qty_sold",
        "理論庫存(份)": "theoretical_stock",
        "實際盤點(份)": "closing_stock",
        "差異/耗損(份)": "variance_qty_raw",
        "差異率": "variance_ratio_legacy",
        "進貨成本單價(元)": "unit_cost",
        "售價(元)": "price",
    })
    df["store_id"] = STORE_ID
    df["on_order_qty"] = 0.0        # 已下單未到貨:目前無此資料(P1-04 待補)
    return df


def load_cost_summary(path=XLSX):
    """成本支出分析 -> dict。原表是「標籤 / 數值」的兩欄格式。"""
    raw = pd.read_excel(path, sheet_name="成本支出分析", header=None)
    out = {}
    for _, r in raw.iterrows():
        k, v = r.iloc[0], r.iloc[1]
        if pd.notna(k) and pd.notna(v) and isinstance(v, (int, float)):
            out[str(k).strip()] = float(v)
    return out


# ------------------------------------------------------------------
# 供應商主檔(缺資料 -> 品類預設值 + 標記待補齊)
# ------------------------------------------------------------------
def build_supplier_master(inv):
    """
    規格書 4.2 例外處理:
      「安全庫存或供應商資料缺失:以品類預設值代入並標記待補齊,
        不阻擋叫貨建議產生。」

    這條規則很重要 —— 導入初期資料一定不齊,如果因為缺一個 MOQ 就不出建議,
    系統第一天就沒人用了。正確做法是給預設值、把缺口列出來、繼續運作。
    """
    rows = []
    for _, r in inv.iterrows():
        d = SUPPLIER_DEFAULTS.get(r["category"], CATEGORY_FALLBACK)
        rows.append({
            "store_id": r["store_id"],
            "sku_id": r["sku_id"],
            "category": r["category"],
            "supplier_id": f"SUP-{r['category']}",
            "unit_price": r["unit_cost"],
            "shelf_life_days": SHELF_LIFE_DAYS.get(r["category"], 5),
            "needs_data_fill": True,     # 全部都是預設值 -> 全部待補齊
            **d,
        })
    return pd.DataFrame(rows)


def service_level_z(service_level):
    """
    服務水準 -> Z 值(規格書 4.3:95% -> Z=1.65)

    用查表而不是 scipy.stats.norm.ppf,是為了讓這個專案不必依賴 scipy,
    且規格書列的就是這幾個常用水準。
    """
    table = {0.90: 1.28, 0.95: 1.65, 0.97: 1.88, 0.98: 2.05, 0.99: 2.33}
    if service_level in table:
        return table[service_level]
    ks = np.array(sorted(table))
    return float(np.interp(service_level, ks, [table[k] for k in ks]))


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR


def banner(title, width=70):
    print("=" * width)
    print(title)
    print("=" * width)


def section(title, width=70):
    print(f"\n── {title} " + "─" * max(0, width - len(title) - 4))
