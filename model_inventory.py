"""
模組五:銷售預測與叫貨建議(含廚餘風險)

這個模組回答店長每天下午都要做一次的決定:
    「今天要叫多少貨?」

叫少了 -> 招牌菜賣完,客人白跑一趟,下次不來
叫多了 -> 生鮮過期變廚餘,錢直接丟掉

所以叫貨這件事不是「憑感覺抓一個量」,而是四個數字的平衡:
    日均銷量  x  前置期  +  安全庫存  -  現有庫存  =  建議叫貨量
                                        ^
                        且不得超過「保存期限內賣得完」的上限

四個關鍵設計(這些是餐飲現場的知識,不是教科書公式)
------------------------------------------------------
1. 日均銷量必須排除公休日
   這家店週一公休。如果直接拿「總銷量 / 天數」,分母多算了 1/7 的零銷售日,
   算出來的日均會低估約 14%,叫貨就會系統性地偏少 —— 這是最常見的錯誤。

2. 必須考慮週幾效應
   餐酒館週五六的來客是平日的 2 倍以上。用同一個日均去備週二和週六,
   結果就是週二剩一堆、週六賣光。所以要用「週幾別日均」。

3. 服務水準要分級,不能一刀切(ABC 分析)
   招牌菜斷貨的代價 ≠ 冷門菜斷貨的代價。
   毛利貢獻前段的品項要備到 98% 不斷貨,尾段的備到 90% 就好。
   這就是為什麼「賣最好的菜反而最容易斷貨」—— 因為大家用同一把尺備量。

4. 保存期限是硬上限,而且是廚餘的來源
   海鮮只能放 2 天,酒水可以放一年。
   所以「安全庫存」對海鮮和對酒水是完全不同的概念:
   酒水可以囤(囤了還能撐過缺貨),生鮮囤了就是廚餘。
   這是餐酒館的結構性優勢 —— 毛利最高的品項剛好最耐放。

輸出
----
  output/reorder_advice.csv     叫貨建議表(給店長的行動清單)
  output/demand_profile.csv     各品項需求特徵(日均/波動/ABC 分級)
"""

import os
import sys

import numpy as np
import pandas as pd

from config import BUSINESS, DATA_DIR, OUTPUT_DIR

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ------------------------------------------------------------------
# 參數:保存期限與前置期
# ------------------------------------------------------------------
# 保存期限(天):從進貨到必須用掉的天數。
# 這是叫貨量的硬上限 —— 叫超過這個量,多的部分注定變廚餘。
SHELF_LIFE = {
    "Seafood":   2,    # 海鮮:當日或隔日必須用完,餐飲業最不能囤的東西
    "Dessert":   3,    # 手作甜點:冷藏保存有限
    "Red_Meat":  4,    # 紅肉:真空冷藏可撐幾天
    "Universal": 7,    # 蔬菜類
    "Cheese":   14,    # 起司/火腿:熟成品,耐放
    "Fried":    30,    # 冷凍炸物
}
DRINK_SHELF_LIFE = 365   # 酒水:實務上視為不會壞

# 前置期(天):今天下單,幾天後到貨。期間的銷售要靠現有庫存撐。
LEAD_TIME = {
    "Seafood":   1,    # 海鮮每日配送
    "Dessert":   1,
    "Red_Meat":  2,
    "Universal": 2,
    "Cheese":    3,
    "Fried":     3,
}
DRINK_LEAD_TIME = 5     # 酒商配送較慢,所以酒水更需要提前叫

# ABC 分級對應的服務水準與安全係數 Z
# 服務水準 = 「不缺貨的機率」。98% 代表 100 次叫貨只容許 2 次缺貨。
SERVICE_LEVEL = {
    "A": (0.98, 2.05),   # 毛利貢獻累積前 70%:招牌菜,斷貨代價最高
    "B": (0.95, 1.65),   # 70% ~ 90%
    "C": (0.90, 1.28),   # 尾段 10%:備太多反而是浪費
}


# ------------------------------------------------------------------
# 步驟一:把訂單明細變成「每日每品項銷量」
# ------------------------------------------------------------------
def load_daily_demand(items):
    """
    order_items.csv 每一列是「某一桌點了某一道菜」,
    所以 groupby(date, item_id).size() 就是那天賣出的份數。

    回傳:長格式 DataFrame(date, item_id, item_name, type, pairing_tag, units)
    """
    daily = (
        items.groupby(["date", "item_id", "item_name", "type", "pairing_tag"])
             .size()
             .reset_index(name="units")
    )
    daily["date"] = pd.to_datetime(daily["date"])
    daily["weekday"] = daily["date"].dt.weekday
    return daily


def _business_days(daily):
    """
    營業日清單(排除公休日)。

    為什麼要這樣做:直接用 date.nunique() 只會算到「有銷售紀錄的日子」,
    看起來剛好排除了公休日 —— 但如果某天某個品項剛好沒賣出去,
    那天對這個品項就消失了,分母又會偏小。
    所以營業日要從「整間店的日期集合」取,不是從單一品項取。
    """
    all_days = pd.Series(sorted(daily["date"].unique()))
    closed = BUSINESS["closed_weekday"]
    return all_days[all_days.dt.weekday != closed]


# ------------------------------------------------------------------
# 步驟二:需求特徵 + ABC 分級
# ------------------------------------------------------------------
def demand_profile(daily, menu):
    """
    算出每個品項的需求特徵:

      avg_daily     營業日平均銷量(已排除公休日)
      std_daily     銷量標準差 -> 波動越大,安全庫存要越多
      peak_daily    週五六的平均銷量 -> 尖峰備量的依據
      abc           依「毛利貢獻」做 ABC 分級,而不是依銷量

    為什麼 ABC 要用毛利而不是銷量:
    松露薯條賣得比戰斧牛排多得多,但一份薯條的毛利是 130 元,
    一份戰斧牛排是 308 元。用銷量分級,會把真正撐起這家店的品項排到後面。
    """
    biz_days = _business_days(daily)
    n_biz = len(biz_days)

    # 攤平成「品項 x 營業日」的完整矩陣,沒賣出的日子補 0
    # 少了這一步,標準差會被高估(因為只看有賣的日子)
    idx = pd.MultiIndex.from_product(
        [daily["item_id"].unique(), biz_days], names=["item_id", "date"]
    )
    wide = (daily.set_index(["item_id", "date"])["units"]
                 .reindex(idx, fill_value=0)
                 .reset_index())

    prof = wide.groupby("item_id")["units"].agg(
        total_units="sum", avg_daily="mean", std_daily="std"
    ).reset_index()

    # 尖峰日(週五六)平均銷量
    wide["weekday"] = wide["date"].dt.weekday
    peak = (wide[wide["weekday"].isin([4, 5])]
            .groupby("item_id")["units"].mean()
            .rename("peak_daily").reset_index())

    prof = prof.merge(peak, on="item_id", how="left")
    prof = prof.merge(
        menu[["item_id", "item_name", "type", "pairing_tag", "price", "profit_margin"]],
        on="item_id", how="left",
    )

    # 毛利貢獻與 ABC 分級
    prof["unit_profit"] = (prof["price"] * prof["profit_margin"]).round(1)
    prof["total_profit"] = (prof["unit_profit"] * prof["total_units"]).round(0)
    prof = prof.sort_values("total_profit", ascending=False).reset_index(drop=True)
    cum_share = prof["total_profit"].cumsum() / prof["total_profit"].sum()
    prof["cum_profit_share"] = cum_share.round(3)
    prof["abc"] = np.where(cum_share <= 0.70, "A",
                  np.where(cum_share <= 0.90, "B", "C"))

    prof["shelf_life"] = prof.apply(_shelf_life, axis=1)
    prof["lead_time"] = prof.apply(_lead_time, axis=1)
    prof["business_days"] = n_biz
    return prof


def _shelf_life(row):
    if row["type"] == "酒水":
        return DRINK_SHELF_LIFE
    return SHELF_LIFE.get(row["pairing_tag"], 7)


def _lead_time(row):
    if row["type"] == "酒水":
        return DRINK_LEAD_TIME
    return LEAD_TIME.get(row["pairing_tag"], 2)


# ------------------------------------------------------------------
# 步驟三:安全庫存與再訂購點
# ------------------------------------------------------------------
def safety_stock(std_daily, lead_time, z):
    """
    安全庫存 = Z x 需求標準差 x sqrt(前置期)

    為什麼要乘 sqrt(前置期) 而不是前置期本身:
    前置期越長,累積的不確定性越大,但不是線性成長 ——
    多天的需求波動會互相抵銷一部分(有的日子多、有的日子少),
    統計上累積標準差是 sqrt(天數) 倍。這是庫存管理的標準結果。
    """
    return z * std_daily * np.sqrt(lead_time)


def reorder_advice(prof, stock, target_weekday=None):
    """
    產生叫貨建議表。

    參數
    ----
    prof            demand_profile() 的輸出
    stock           現有庫存,dict {item_id: 份數}
                    ** 庫存必須由店長盤點輸入,系統不該假裝自己知道。 **
                    這是系統導入時最常出錯的地方:
                    以為串了 POS 就能自動算庫存,但 POS 只知道「賣掉幾份」,
                    不知道試菜、員工餐、掉在地上、切壞的那幾份。
                    庫存永遠要以實際盤點為準。
    target_weekday  要備到哪一天(0=Mon ... 6=Sun)。
                    None 表示用整體平均;指定週五六會得到尖峰備量。

    回傳:每個品項一列的建議表,已依急迫度排序
    """
    rows = []
    for _, r in prof.iterrows():
        z = SERVICE_LEVEL[r["abc"]][1]
        service = SERVICE_LEVEL[r["abc"]][0]

        # 要備到哪個需求水準:尖峰日用 peak_daily,平日用 avg_daily
        base_daily = (r["peak_daily"] if target_weekday in (4, 5)
                      else r["avg_daily"])

        ss = safety_stock(r["std_daily"], r["lead_time"], z)
        # 再訂購點:前置期內會賣掉的量 + 安全庫存
        rop = base_daily * r["lead_time"] + ss

        on_hand = float(stock.get(r["item_id"], 0))
        raw_need = rop - on_hand

        # ---- 保存期限上限:叫再多也只能叫到「保存期內賣得完」的量 ----
        # 這一行就是廚餘控制的核心。少了它,系統會建議把海鮮囤成安全庫存。
        max_by_shelf = base_daily * r["shelf_life"]
        suggest = max(0.0, min(raw_need, max_by_shelf - on_hand))

        # 缺貨風險:現有庫存撐得過前置期嗎?
        days_of_cover = on_hand / base_daily if base_daily > 0 else np.inf
        if days_of_cover < r["lead_time"]:
            status = "🔴 立即叫貨"      # 到貨前就會賣完
        elif on_hand < rop:
            status = "🟡 低於安全庫存"
        else:
            status = "🟢 庫存充足"

        # ---- 廚餘風險:現有庫存在保存期內賣不完的部分 ----
        sellable = base_daily * r["shelf_life"]
        excess = max(0.0, on_hand - sellable)
        waste_cost = excess * r["price"] * (1 - r["profit_margin"])   # 食材成本

        rows.append({
            "item_id": r["item_id"],
            "品項": r["item_name"],
            "類別": r["type"],
            "ABC": r["abc"],
            "服務水準": f"{service:.0%}",
            "日均銷量": round(base_daily, 1),
            "需求波動": round(r["std_daily"], 1),
            "前置期(天)": int(r["lead_time"]),
            "保存期(天)": int(r["shelf_life"]),
            "安全庫存": round(ss, 1),
            "再訂購點": round(rop, 1),
            "現有庫存": round(on_hand, 1),
            "可撐天數": round(days_of_cover, 1) if np.isfinite(days_of_cover) else None,
            "建議叫貨": int(np.ceil(suggest)),
            "狀態": status,
            "過剩份數": round(excess, 1),
            "廚餘成本": round(waste_cost, 0),
        })

    out = pd.DataFrame(rows)
    # 排序:先看急迫度,同級再看毛利重要性
    order = {"🔴 立即叫貨": 0, "🟡 低於安全庫存": 1, "🟢 庫存充足": 2}
    out["_s"] = out["狀態"].map(order)
    out["_a"] = out["ABC"].map({"A": 0, "B": 1, "C": 2})
    return out.sort_values(["_s", "_a"]).drop(columns=["_s", "_a"]).reset_index(drop=True)


# ------------------------------------------------------------------
# 產生一份「今日盤點快照」用來示範
# ------------------------------------------------------------------
def simulate_stock_snapshot(prof, seed=42):
    """
    模擬店長今天下午盤點的結果。

    這個函式改過一次,原因值得記下來 ——
    第一版我不分品類、一律讓熱銷品只剩 0.4~1.2 天的量,結果 28 個品項裡有 24 個
    被標成「立即叫貨」。**全部都紅等於沒有紅**,店長看一次就不會再看第二次。
    這就是「提醒疲勞」(alert fatigue),也是導入的系統上線一個月就被棄用的頭號原因。

    根因不在演算法,在假設不合現場:
    餐酒館的酒是**按箱囤**的(前置期 5 天、又不會壞),庫存本來就有好幾週;
    只有生鮮才會被盤到剩不到一天。庫存水位本來就該依「保存期限」分層。

    所以現在按保存期分三層,層內再讓熱銷品比冷門品更緊:
      生鮮(<=4 天)   每日進貨,水位低     -> 斷貨風險集中在這裡
      中期(5~30 天)  週配,水位中等
      酒水(365 天)   按箱囤,水位高       -> 幾乎不會斷,但會壓資金

    刻意保留兩個真實現象:
      - 賣最好的品項最容易被盤到低水位(消耗快、補得慢)  <- 斷貨的來源
      - 冷門品項反而囤了一堆(當初怕不夠,結果賣不動)    <- 廚餘的來源
    """
    rng = np.random.default_rng(seed)
    # (保存期上限, A/B 級庫存天數區間, C 級庫存天數區間)
    BANDS = [
        (4,   (0.6, 2.0), (2.5, 5.0)),      # 生鮮
        (30,  (3.0, 6.0), (6.0, 12.0)),     # 中期
        (999, (12.0, 25.0), (25.0, 45.0)),  # 酒水
    ]

    stock = {}
    for _, r in prof.iterrows():
        for limit, hot, cold in BANDS:
            if r["shelf_life"] <= limit:
                lo, hi = cold if r["abc"] == "C" else hot
                break
        days = rng.uniform(lo, hi)
        stock[r["item_id"]] = round(r["avg_daily"] * days)
    return stock


# ------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------
def main():
    data_dir = os.path.join(BASE_DIR, DATA_DIR)
    out_dir = os.path.join(BASE_DIR, OUTPUT_DIR)
    os.makedirs(out_dir, exist_ok=True)

    items = pd.read_csv(os.path.join(data_dir, "order_items.csv"))
    menu = pd.read_csv(os.path.join(data_dir, "menu.csv"))

    daily = load_daily_demand(items)
    prof = demand_profile(daily, menu)

    biz_days = int(prof["business_days"].iloc[0])
    span = f"{daily['date'].min():%Y-%m-%d} ~ {daily['date'].max():%Y-%m-%d}"

    print("=" * 66)
    print("餐酒館 銷售預測與叫貨建議")
    print("=" * 66)
    print(f"分析期間 {span}(營業日 {biz_days} 天,已排除週一公休)")
    print(f"品項數 {len(prof)}   訂單明細 {len(items):,} 筆\n")

    print("── ABC 分級(依毛利貢獻)" + "─" * 40)
    for g in ["A", "B", "C"]:
        sub = prof[prof["abc"] == g]
        share = sub["total_profit"].sum() / prof["total_profit"].sum()
        sl = SERVICE_LEVEL[g][0]
        print(f"  {g} 級  {len(sub):>2} 品項   毛利佔比 {share:>5.1%}   "
              f"服務水準 {sl:.0%}")
    print()

    # 盤點快照 -> 叫貨建議
    stock = simulate_stock_snapshot(prof)
    advice = reorder_advice(prof, stock, target_weekday=4)   # 備到週五尖峰

    urgent = advice[advice["狀態"] == "🔴 立即叫貨"]
    waste = advice[advice["過剩份數"] > 0].sort_values("廚餘成本", ascending=False)

    print("── 🔴 需要立即叫貨(到貨前就會賣完)" + "─" * 26)
    if urgent.empty:
        print("  無\n")
    else:
        cols = ["品項", "ABC", "日均銷量", "現有庫存", "可撐天數",
                "前置期(天)", "建議叫貨"]
        print(urgent[cols].to_string(index=False))
        print()

    print("── ⚠️ 廚餘風險(保存期內賣不完)" + "─" * 30)
    if waste.empty:
        print("  無\n")
    else:
        cols = ["品項", "ABC", "日均銷量", "保存期(天)", "現有庫存",
                "過剩份數", "廚餘成本"]
        print(waste[cols].head(8).to_string(index=False))
        print(f"\n  合計廚餘成本 {waste['廚餘成本'].sum():,.0f} 元"
              f"(單次盤點,非月累計)")
        print()

    print("── 摘要" + "─" * 56)
    print(f"  立即叫貨      {len(urgent)} 項")
    print(f"  低於安全庫存  {(advice['狀態'] == '🟡 低於安全庫存').sum()} 項")
    print(f"  庫存充足      {(advice['狀態'] == '🟢 庫存充足').sum()} 項")
    print(f"  建議叫貨總份數 {advice['建議叫貨'].sum():,} 份")
    print()

    advice.to_csv(os.path.join(out_dir, "reorder_advice.csv"),
                  index=False, encoding="utf-8-sig")
    prof.to_csv(os.path.join(out_dir, "demand_profile.csv"),
                index=False, encoding="utf-8-sig")
    print(f"已輸出 {OUTPUT_DIR}/reorder_advice.csv")
    print(f"已輸出 {OUTPUT_DIR}/demand_profile.csv")

    return prof, advice


if __name__ == "__main__":
    main()
