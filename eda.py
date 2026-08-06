"""
模組零:營運現況分析(EDA)

這是整個專題的第一步 —— 在建任何模型之前,先看懂這家店在發生什麼事。

面試時的順序也應該是這樣:先講「我從資料看到什麼營運問題」,
再講「所以我建了什麼模型」。反過來先講模型,聽起來就會像在炫技。

輸出六張圖(output/eda_*.png)與一份 KPI 摘要。
每張圖都對應一個營運問題,不是為了畫圖而畫圖。
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE_DIR, "data")
OUT = os.path.join(BASE_DIR, "output")

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["axes.unicode_minus"] = False
# 圖表標題一律用英文,確保在沒裝中文字型的環境也不會變成方框

WEEKDAY_EN = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}


def kpi_summary(visits, items):
    """列出餐酒館最該看的幾個數字。"""
    n_days = visits["date"].nunique()
    total_rev = visits["total_revenue"].sum()
    total_profit = visits["total_profit"].sum()

    print("=" * 70)
    print("營運 KPI 摘要")
    print("=" * 70)
    print(f"營業天數          {n_days} 天")
    print(f"總桌數            {len(visits):,} 桌")
    print(f"日均桌數          {len(visits) / n_days:.1f} 桌")
    print(f"總營收            {total_rev:,.0f} 元")
    print(f"日均營收          {total_rev / n_days:,.0f} 元")
    print(f"總毛利            {total_profit:,.0f} 元(毛利率 {total_profit / total_rev:.1%})")
    print()
    print(f"平均客單價        {visits['spend_per_head'].mean():.0f} 元/人")
    print(f"平均每桌營收      {visits['total_revenue'].mean():.0f} 元")
    print(f"平均同桌人數      {visits['party_size'].mean():.2f} 人")
    print(f"平均停留時間      {visits['duration_min'].mean():.0f} 分鐘")
    print()
    print("--- 餐酒館核心指標 ---")
    print(f"酒水佔比          {visits['drink_revenue'].sum() / total_rev:.1%}"
          f"  (業界目標 30~40%)")
    print(f"人均酒水杯數      {(visits['n_drink'] / visits['party_size']).mean():.2f} 杯")
    print(f"零酒水桌比例      {(visits['n_drink'] == 0).mean():.1%}"
          f"  <- 這些桌是流失的毛利")

    # 酒水 vs 餐點的毛利貢獻對比:餐酒館的獲利結構
    drink_items = items[items["type"] == "酒水"]
    food_items = items[items["type"] != "酒水"]
    print()
    print("--- 毛利結構(這是餐酒館與一般餐廳最大的差異)---")
    print(f"餐點:營收佔 {food_items['price'].sum() / total_rev:.1%}"
          f",毛利佔 {food_items['profit'].sum() / total_profit:.1%}")
    print(f"酒水:營收佔 {drink_items['price'].sum() / total_rev:.1%}"
          f",毛利佔 {drink_items['profit'].sum() / total_profit:.1%}")
    print("  -> 酒水的毛利佔比高於營收佔比,證明它是獲利槓桿而非附屬品")


def plot_all(visits, items):
    os.makedirs(OUT, exist_ok=True)

    # --- 圖 1:到店時段(驗證雙峰結構)---
    fig, ax = plt.subplots(figsize=(10, 5))
    hourly = visits.groupby("arrival_hour").size()
    ax.bar(hourly.index, hourly.values, color="#4C72B0")
    ax.axvline(21, color="crimson", ls="--", lw=1.4)
    ax.text(21.1, hourly.max() * 0.92, "Buzz hour starts", color="crimson", fontsize=9)
    ax.set_xlabel("Arrival Hour")
    ax.set_ylabel("Tables")
    ax.set_title("Arrival Time Distribution - Dinner Peak + Buzz Peak")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "eda_1_arrival_hour.png"), dpi=120)
    plt.close()

    # --- 圖 2:週幾營收 ---
    fig, ax = plt.subplots(figsize=(10, 5))
    wd = visits.groupby("weekday")["total_revenue"].sum()
    ax.bar([WEEKDAY_EN[i] for i in wd.index], wd.values, color="#55A868")
    ax.set_ylabel("Revenue (NTD)")
    ax.set_title("Revenue by Weekday (Mon closed)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "eda_2_weekday.png"), dpi=120)
    plt.close()

    # --- 圖 3:酒水佔比分布(對照 40% 目標線)---
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(visits["drink_ratio"], bins=30, kde=True, ax=ax, color="#8172B2")
    ax.axvline(0.40, color="crimson", ls="--", lw=1.5)
    ax.text(0.41, ax.get_ylim()[1] * 0.9, "Target 40%", color="crimson", fontsize=9)
    ax.set_xlabel("Drink Revenue Ratio")
    ax.set_title("Drink Ratio Distribution - The Core Bistro KPI")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "eda_3_drink_ratio.png"), dpi=120)
    plt.close()

    # --- 圖 4:停留時間 vs 酒水杯數(餐酒館的獲利機制)---
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = [0, 60, 90, 120, 150, 300]
    labels = ["<60", "60-90", "90-120", "120-150", ">150"]
    tmp = visits.copy()
    tmp["dur_bin"] = pd.cut(tmp["duration_min"], bins=bins, labels=labels)
    grp = tmp.groupby("dur_bin", observed=True).apply(
        lambda g: (g["n_drink"] / g["party_size"]).mean(), include_groups=False
    )
    ax.bar(grp.index.astype(str), grp.values, color="#C44E52")
    ax.set_xlabel("Duration (minutes)")
    ax.set_ylabel("Drinks per Person")
    ax.set_title("Longer Stay = More Drinks (Why Table Turnover Is Not the Bistro KPI)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "eda_4_duration_vs_drinks.png"), dpi=120)
    plt.close()

    # --- 圖 5:品項毛利貢獻(柏拉圖)---
    fig, ax = plt.subplots(figsize=(11, 6))
    prof = (items.groupby("item_name")["profit"].sum()
                 .sort_values(ascending=False).head(15))
    ax.barh(range(len(prof)), prof.values[::-1], color="#4C72B0")
    ax.set_yticks(range(len(prof)))
    ax.set_yticklabels([f"#{i+1}" for i in range(len(prof))][::-1])
    ax.set_xlabel("Total Profit (NTD)")
    ax.set_title("Top 15 Items by Profit Contribution")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "eda_5_item_profit.png"), dpi=120)
    plt.close()

    # --- 圖 6:各時段酒水佔比(找出該加強推酒的時段)---
    fig, ax = plt.subplots(figsize=(10, 5))
    hr = visits.groupby("arrival_hour")["drink_ratio"].mean()
    ax.plot(hr.index, hr.values, marker="o", color="#8172B2", lw=2)
    ax.axhline(0.40, color="crimson", ls="--", lw=1.2)
    ax.set_xlabel("Arrival Hour")
    ax.set_ylabel("Avg Drink Ratio")
    ax.set_title("Drink Ratio by Arrival Hour - Where to Push Drinks")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "eda_6_ratio_by_hour.png"), dpi=120)
    plt.close()

    print(f"\n六張圖已輸出到 {OUT}/")


def operational_findings(visits, items):
    """把數據轉成營運可以行動的結論 —— 這才是營運職要交出的東西。"""
    print("\n" + "=" * 70)
    print("營運發現與建議")
    print("=" * 70)

    # 1. 最弱時段
    hr = visits.groupby("arrival_hour").agg(
        桌數=("visit_id", "count"), 酒水佔比=("drink_ratio", "mean"))
    weak = hr[hr["桌數"] >= 30].sort_values("酒水佔比").head(2)
    print("\n[1] 酒水佔比最低的時段")
    for h, r in weak.iterrows():
        print(f"    {h}:00  酒水佔比僅 {r['酒水佔比']:.1%}({int(r['桌數'])} 桌)")
    print("    建議:這幾個時段推『餐前酒優惠』,把佔比拉近 40% 目標線")

    # 2. 零酒水桌
    zero = visits[visits["n_drink"] == 0]
    if len(zero):
        print(f"\n[2] 零酒水桌共 {len(zero)} 桌({len(zero)/len(visits):.1%})")
        print(f"    這些桌平均客單價僅 {zero['spend_per_head'].mean():.0f} 元,"
              f"低於整體 {visits['spend_per_head'].mean():.0f} 元")
        lost = len(zero) * visits[visits.n_drink > 0]["drink_revenue"].mean()
        print(f"    若其中一半願意點一杯,估計可增加營收 {lost * 0.5:,.0f} 元")

    # 3. 平日 vs 週末
    wk = visits.groupby("is_weekend").agg(
        日均桌數=("visit_id", "count"), 客單價=("spend_per_head", "mean"),
        酒水佔比=("drink_ratio", "mean"))
    print("\n[3] 平日 vs 週五六")
    for k, r in wk.iterrows():
        tag = "週五六" if k else "平日"
        print(f"    {tag}:客單價 {r['客單價']:.0f} 元,酒水佔比 {r['酒水佔比']:.1%}")
    print("    注意:若兩者客單價相近,表示週末只是『人多』而非『每桌消費更高』")
    print("          -> 平日的成長空間在提高來客,週末的空間在提高單桌消費")

    # 4. 高毛利但賣不好的品項(菜單優化機會)
    stat = items.groupby("item_name").agg(
        銷量=("item_id", "count"),
        單品毛利=("profit", "mean"),
        總毛利=("profit", "sum")).reset_index()
    stat["銷量分位"] = stat["銷量"].rank(pct=True)
    stat["毛利分位"] = stat["單品毛利"].rank(pct=True)
    hidden = stat[(stat["毛利分位"] > 0.6) & (stat["銷量分位"] < 0.4)]
    print("\n[4] 高毛利但銷量偏低的品項(菜單優化機會)")
    if hidden.empty:
        print("    無明顯品項")
    else:
        for _, r in hidden.sort_values("單品毛利", ascending=False).iterrows():
            print(f"    {r['item_name']:20s} 單品毛利 {r['單品毛利']:.0f} 元,"
                  f"僅賣出 {int(r['銷量'])} 份")
        print("    建議:這些品項調整菜單位置、加註推薦標記,或納入外場主動推薦話術")


def main():
    visits = pd.read_csv(os.path.join(DATA, "visits.csv"))
    items = pd.read_csv(os.path.join(DATA, "order_items.csv"))

    kpi_summary(visits, items)
    plot_all(visits, items)
    operational_findings(visits, items)


if __name__ == "__main__":
    main()
