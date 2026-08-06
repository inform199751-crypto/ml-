"""
模組一:購物籃關聯分析 + 毛利感知推薦  (專題核心)

商業問題
--------
「客人點了舒肥紅酒燉牛肉,外場該推哪一杯酒?」

一般推薦系統的做法
------------------
用關聯規則算出「最可能被一起點」的品項(confidence 最高),然後推那個。
問題是:最可能被加點的,往往是最便宜、毛利最低的品項。
推薦成功了,但沒賺到錢。

這個專題的做法(差異化重點)
--------------------------
把「接受機率」和「毛利」放在一起看,改用**期望毛利**排序:

    期望毛利 = confidence(信賴度) x 售價 x 毛利率

舉例:
    A 酒:接受機率 60%,毛利 100 元  ->  期望毛利 60 元
    B 酒:接受機率 40%,毛利 266 元  ->  期望毛利 106 元
    結論:該推 B,雖然它「比較不容易成功」

這個轉換就是餐飲營運的思維 —— 追求的不是推薦成功率,而是每桌毛利。
純技術背景的人做推薦系統時,幾乎不會想到要把毛利放進排序函數。

演算法說明
----------
這裡手寫 Apriori 的核心指標(support / confidence / lift),沒有用套件,
因為專題面試會被問「這個數字怎麼來的」,手寫才講得清楚:

    support(A)     A 出現在多少比例的訂單裡        -> 這道菜紅不紅
    confidence(A→B) 點了 A 的訂單裡,有多少也點了 B -> 加點成功率
    lift(A→B)      confidence / support(B)         -> 是真的相關,還是 B 本來就熱賣
                   lift > 1 才是有意義的關聯
"""

import os
import sys
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE_DIR, "data")


# ------------------------------------------------------------------
# 1. 關聯規則計算
# ------------------------------------------------------------------
def build_baskets(items_df, only_hour=None):
    """
    把品項明細轉成「一張訂單一個品項集合」。

    only_hour: 若指定 (start, end),只取該時段的訂單
               -> 這就是「餐期感知」:晚餐時段與微醺時段的搭配習慣不一樣,
                  分開算才能給出對應時段的建議
    """
    df = items_df
    if only_hour is not None:
        visits = pd.read_csv(os.path.join(DATA, "visits.csv"))
        lo, hi = only_hour
        keep = visits.loc[visits["arrival_hour"].between(lo, hi), "visit_id"]
        df = df[df["visit_id"].isin(set(keep))]
    return df.groupby("visit_id")["item_id"].apply(set)


def association_rules(baskets, menu, min_support=0.01, min_confidence=0.05,
                      antecedent_types=("主食", "副食"), consequent_types=("酒水",)):
    """
    計算 A -> B 的關聯規則。

    預設只算「餐點 -> 酒水」的方向,因為這才是我們的商業問題
    (外場要決定的是「點了菜之後推什麼酒」,不是反過來)。
    """
    n = len(baskets)
    menu_idx = menu.set_index("item_id")

    ante_ids = set(menu.loc[menu["type"].isin(antecedent_types), "item_id"])
    cons_ids = set(menu.loc[menu["type"].isin(consequent_types), "item_id"])

    # --- 單品 support ---
    single = {}
    for basket in baskets:
        for iid in basket:
            single[iid] = single.get(iid, 0) + 1
    support = {k: v / n for k, v in single.items()}

    # --- 配對計數(只數我們關心的方向) ---
    pair = {}
    for basket in baskets:
        a_items = basket & ante_ids
        b_items = basket & cons_ids
        for a in a_items:
            for b in b_items:
                pair[(a, b)] = pair.get((a, b), 0) + 1

    rows = []
    for (a, b), cnt in pair.items():
        sup_ab = cnt / n
        if sup_ab < min_support:
            continue
        conf = sup_ab / support[a]
        if conf < min_confidence:
            continue
        lift = conf / support[b]

        b_row = menu_idx.loc[b]
        unit_profit = float(b_row["price"]) * float(b_row["profit_margin"])

        rows.append({
            "antecedent": menu_idx.loc[a, "item_name"],
            "antecedent_id": a,
            "consequent": b_row["item_name"],
            "consequent_id": b,
            "support": round(sup_ab, 4),
            "confidence": round(conf, 4),
            "lift": round(lift, 3),
            "unit_profit": round(unit_profit, 1),
            # 核心指標:期望毛利 = 加點成功率 x 單杯毛利
            "expected_profit": round(conf * unit_profit, 1),
        })

    return pd.DataFrame(rows).sort_values("expected_profit", ascending=False)


# ------------------------------------------------------------------
# 2. 推薦介面(供 Streamlit 點餐機呼叫)
# ------------------------------------------------------------------
def recommend(rules, ordered_item_ids, top_n=3, strategy="profit"):
    """
    依照客人已點的餐點,推薦酒水。

    strategy:
      "profit"     依期望毛利排序   <- 本專題主張的做法
      "confidence" 依加點成功率排序  <- 一般推薦系統做法(用來對照)
    """
    hits = rules[rules["antecedent_id"].isin(ordered_item_ids)]
    if hits.empty:
        return pd.DataFrame()

    sort_key = "expected_profit" if strategy == "profit" else "confidence"

    # 同一杯酒可能被多道菜關聯到,取該酒最好的那一條規則
    best = (hits.sort_values(sort_key, ascending=False)
                .drop_duplicates("consequent_id")
                .head(top_n))
    return best[["consequent", "confidence", "unit_profit", "expected_profit", "lift", "antecedent"]]


# ------------------------------------------------------------------
# 3. 主流程
# ------------------------------------------------------------------
def main():
    items = pd.read_csv(os.path.join(DATA, "order_items.csv"))
    menu = pd.read_csv(os.path.join(DATA, "menu.csv"))

    baskets = build_baskets(items)
    rules = association_rules(baskets, menu)

    print("=" * 74)
    print("餐點 -> 酒水 關聯規則(依期望毛利排序,前 15 條)")
    print("=" * 74)
    show = rules.head(15).copy()
    show["confidence"] = (show["confidence"] * 100).round(1).astype(str) + "%"
    print(show[["antecedent", "consequent", "confidence",
                "lift", "unit_profit", "expected_profit"]].to_string(index=False))

    # ---- 對照實驗:兩種排序策略推出來的酒不一樣 ----
    print("\n" + "=" * 74)
    print("對照實驗:同一道菜,兩種推薦策略的差異")
    print("=" * 74)

    demo_id = "M01"   # 舒肥紅酒燉牛肉
    demo_name = menu.loc[menu["item_id"] == demo_id, "item_name"].iloc[0]
    print(f"\n情境:客人點了「{demo_name}」\n")

    for strat, label in [("confidence", "A. 一般做法:推最容易被接受的"),
                         ("profit", "B. 本專題做法:推期望毛利最高的")]:
        rec = recommend(rules, [demo_id], top_n=3, strategy=strat)
        print(label)
        if rec.empty:
            print("  (無規則)")
            continue
        for _, r in rec.iterrows():
            print(f"  {r['consequent']:20s} 成功率 {r['confidence']*100:5.1f}%"
                  f"  單杯毛利 {r['unit_profit']:6.0f}"
                  f"  期望毛利 {r['expected_profit']:6.1f}")
        print(f"  -> 合計期望毛利 {rec['expected_profit'].sum():.1f} 元\n")

    # ---- 量化整體效益:換排序策略能多賺多少 ----
    food_ids = menu.loc[menu["type"].isin(["主食", "副食"]), "item_id"]
    gain_p, gain_c = [], []
    for fid in food_ids:
        rp = recommend(rules, [fid], top_n=3, strategy="profit")
        rc = recommend(rules, [fid], top_n=3, strategy="confidence")
        if not rp.empty:
            gain_p.append(rp["expected_profit"].sum())
        if not rc.empty:
            gain_c.append(rc["expected_profit"].sum())

    if gain_p and gain_c:
        avg_p = sum(gain_p) / len(gain_p)
        avg_c = sum(gain_c) / len(gain_c)
        print("=" * 74)
        print("整體效益評估(對每道餐點各推 3 杯酒,平均期望毛利)")
        print("=" * 74)
        print(f"  依成功率排序:{avg_c:6.1f} 元/桌")
        print(f"  依期望毛利排序:{avg_p:6.1f} 元/桌")
        print(f"  提升:{(avg_p / avg_c - 1) * 100:+.1f}%")

    # ---- 商業規則層:擋掉餐飲上錯誤的搭配 ----
    from business_rules import recommend_with_rules

    print("\n" + "=" * 74)
    print("商業規則層:為什麼需要餐飲知識把關")
    print("=" * 74)

    fish_id = "M06"   # 香煎鮭魚佐檸檬
    fish_name = menu.loc[menu["item_id"] == fish_id, "item_name"].iloc[0]
    print(f"\n情境:客人點了「{fish_name}」(海鮮)\n")

    raw, _ = recommend_with_rules(rules, menu, [fish_id], arrival_hour=20, use_rules=False)
    print("純模型輸出(只看期望毛利):")
    for _, r in raw.iterrows():
        print(f"  {r['consequent']:20s} 期望毛利 {r['expected_profit']:6.1f}")

    fixed, blocked = recommend_with_rules(rules, menu, [fish_id], arrival_hour=20, use_rules=True)
    print("\n套用餐飲規則後:")
    for _, r in fixed.iterrows():
        print(f"  {r['consequent']:20s} 期望毛利 {r['expected_profit']:6.1f}"
              f"  最終分數 {r['final_score']:6.1f}")

    if blocked:
        print("\n被擋掉的搭配與原因(這是純技術背景寫不出來的部分):")
        for b in blocked:
            print(f"  x {b['被擋的酒']:18s} <- {b['原因']}")

    # ---- 餐期感知:晚餐 vs 微醺時段的規則不同 ----
    print("\n" + "=" * 74)
    print("餐期感知:同一道菜在不同時段,該推的酒不一樣")
    print("=" * 74)
    for label, window in [("晚餐時段 17-20 點", (17, 20)), ("微醺時段 21-24 點", (21, 24))]:
        b = build_baskets(items, only_hour=window)
        r = association_rules(b, menu)
        rec = recommend(r, [demo_id], top_n=3)
        print(f"\n{label}(共 {len(b):,} 桌)")
        if rec.empty:
            print("  (資料不足)")
            continue
        for _, row in rec.iterrows():
            print(f"  {row['consequent']:20s} 成功率 {row['confidence']*100:5.1f}%"
                  f"  期望毛利 {row['expected_profit']:6.1f}")

    out = os.path.join(BASE_DIR, "output")
    os.makedirs(out, exist_ok=True)
    rules.to_csv(os.path.join(out, "association_rules.csv"),
                 index=False, encoding="utf-8-sig")
    print(f"\n規則已輸出:output/association_rules.csv({len(rules)} 條)")


if __name__ == "__main__":
    main()
