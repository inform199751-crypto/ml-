"""
模組二:餐飲商業邏輯層(Business Rules)

為什麼需要這一層?
------------------
純粹用資料算出來的推薦會出錯。實測就發生過:
模型算出「香煎鮭魚佐檸檬」該配「Negroni」,因為 Negroni 毛利最高(266 元),
統計上又剛好常被一起點。

但任何在餐酒館做過外場的人都知道這是錯的 ——
Negroni 是苦艾酒 + 金巴利的重口味調酒,會直接蓋掉鮭魚的細緻風味,
而且酒體的苦味會把海鮮的腥味帶出來。這杯酒推出去,客人喝一口就不喝了,
下次不會再來。

模型看得到「一起被點」,看不到「客人喝完的感受」。
這一層就是把餐飲專業知識寫成規則,替模型把關。

這也是整個專題最難被取代的部分:
資料科學能力可以學,但「知道 Negroni 不能配鮭魚」來自實際在場的經驗。

三條規則
--------
  R1 搭配禁忌:風味衝突的組合直接排除或降權
  R2 推薦多樣性:三個推薦不能同類型,要讓客人有選擇(酒/啤酒/無酒精)
  R3 時段調整:越晚越推調酒,收攤前不推需要慢慢喝的紅酒
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ------------------------------------------------------------------
# R1 搭配禁忌表
# ------------------------------------------------------------------
# key   = 餐點的 pairing_tag
# value = 該餐點「不該搭」的酒款 item_id,以及原因
#
# 這張表就是餐飲知識的具體化,每一條都有實際理由,面試時要能講出來。
TABOO = {
    "Seafood": {
        "W06": "Negroni 的金巴利苦味會蓋掉海鮮鮮甜,並帶出腥味",
        "W07": "Old Fashioned 的橡木桶與糖漿風味過重,壓過白肉魚",
        "W02": "卡本內蘇維翁單寧太強,與海鮮蛋白質作用會產生金屬味",
        "W11": "威士忌基調與海鮮衝突",
    },
    "Dessert": {
        "W09": "IPA 的苦度與甜點互相打架",
        "W10": "小麥啤酒的酵母味不適合搭甜點",
        "W02": "乾型紅酒配甜點會顯得更酸澀",
    },
    "Red_Meat": {
        # 紅肉配白酒不是禁忌,只是不理想 -> 用降權而非排除
    },
}

# 不到禁忌程度、但搭配不理想的組合(乘上一個折扣係數)
SOFT_PENALTY = {
    ("Red_Meat", "Seafood"): 0.75,   # 紅肉搭白酒:酒體撐不住油脂
    ("Seafood", "Red_Meat"): 0.70,   # 海鮮搭紅酒:上面已列部分禁忌,其餘降權
}


def apply_taboo(rules, menu):
    """
    R1:套用搭配禁忌。
    回傳過濾後的規則表,並記錄每一筆被擋掉的原因(方便專題展示與說明)。
    """
    menu_idx = menu.set_index("item_id")
    keep_mask, penalties, blocked = [], [], []

    for _, r in rules.iterrows():
        food_tag = menu_idx.loc[r["antecedent_id"], "pairing_tag"]
        drink_id = r["consequent_id"]
        drink_tag = menu_idx.loc[drink_id, "pairing_tag"]

        # 硬禁忌 -> 直接排除
        reason = TABOO.get(food_tag, {}).get(drink_id)
        if reason:
            keep_mask.append(False)
            penalties.append(0.0)
            blocked.append({
                "餐點": r["antecedent"],
                "被擋的酒": r["consequent"],
                "原因": reason,
            })
            continue

        # 軟性降權
        keep_mask.append(True)
        penalties.append(SOFT_PENALTY.get((food_tag, drink_tag), 1.0))

    out = rules[keep_mask].copy()
    out["pairing_factor"] = [p for p in penalties if p > 0]
    # 最終排序分數:期望毛利 x 搭配合適度
    out["final_score"] = (out["expected_profit"] * out["pairing_factor"]).round(1)
    return out.sort_values("final_score", ascending=False), blocked


# ------------------------------------------------------------------
# R3 時段調整
# ------------------------------------------------------------------
def hour_factor(drink_type_tag, drink_id, arrival_hour):
    """
    R3:依時段調整權重。

      21 點後(微醺時段):客人是來喝的,調酒接受度高 -> 調酒加權
      23 點後(收攤前)  :不推需要慢慢品的紅酒,推可以快喝完的 -> 紅酒降權
    """
    cocktails = {"W06", "W07", "W08", "W11", "W12"}
    wines = {"W01", "W02", "W03", "W04"}

    f = 1.0
    if arrival_hour >= 21 and drink_id in cocktails:
        f *= 1.20
    if arrival_hour >= 23 and drink_id in wines:
        f *= 0.70
    return f


# ------------------------------------------------------------------
# R2 推薦多樣性
# ------------------------------------------------------------------
def diversify(candidates, menu, top_n=3):
    """
    R2:確保推薦有多樣性。

    為什麼重要:三杯都推調酒的話,不喝烈酒的客人一杯都不會點,轉換率反而是 0。
    外場實際的做法是給「一支酒 + 一支啤酒 + 一個無酒精」這種組合,
    讓桌上每個人都有東西可以點。

    做法:每個酒種(紅酒/白酒/調酒/啤酒/無酒精)最多取一款,取完再補滿。
    """
    menu_idx = menu.set_index("item_id")

    def category(iid):
        name = str(menu_idx.loc[iid, "item_name"])
        if "無酒精" in name:
            return "無酒精"
        if iid in {"W09", "W10"}:
            return "啤酒"
        if iid in {"W06", "W07", "W08", "W11", "W12"}:
            return "調酒"
        if iid in {"W01", "W02", "W07"}:
            return "紅酒"
        return "白酒/氣泡"

    picked, used_cat = [], set()
    for _, row in candidates.iterrows():
        cat = category(row["consequent_id"])
        if cat in used_cat:
            continue
        used_cat.add(cat)
        picked.append(row)
        if len(picked) >= top_n:
            break

    # 類別不足時,用剩下分數最高的補滿
    if len(picked) < top_n:
        chosen_ids = {r["consequent_id"] for r in picked}
        for _, row in candidates.iterrows():
            if row["consequent_id"] in chosen_ids:
                continue
            picked.append(row)
            if len(picked) >= top_n:
                break

    import pandas as pd
    return pd.DataFrame(picked)


# ------------------------------------------------------------------
# 整合:完整推薦流程
# ------------------------------------------------------------------
def recommend_with_rules(rules, menu, ordered_item_ids, arrival_hour=20,
                         top_n=3, use_rules=True):
    """
    完整的推薦流程(這就是專題的「兩階段 + 商業重排」架構):

      階段一 召回:找出所有與已點餐點相關的酒款(關聯規則)
      階段二 排序:依期望毛利排序(毛利感知)
      階段三 重排:套用餐飲商業規則(禁忌 / 時段 / 多樣性)  <- 差異化在這

    use_rules=False 時跳過階段三,用來做 A/B 對照。
    """
    hits = rules[rules["antecedent_id"].isin(ordered_item_ids)].copy()
    if hits.empty:
        import pandas as pd
        return pd.DataFrame(), []

    blocked = []
    if use_rules:
        hits, blocked = apply_taboo(hits, menu)
        if hits.empty:
            import pandas as pd
            return pd.DataFrame(), blocked
        hits["hour_factor"] = [
            hour_factor(None, iid, arrival_hour) for iid in hits["consequent_id"]
        ]
        hits["final_score"] = (hits["final_score"] * hits["hour_factor"]).round(1)
        hits = hits.sort_values("final_score", ascending=False)
    else:
        hits["final_score"] = hits["expected_profit"]
        hits = hits.sort_values("final_score", ascending=False)

    # 同一款酒可能被多道菜關聯到,只留最好的那條規則
    hits = hits.drop_duplicates("consequent_id")

    result = diversify(hits, menu, top_n=top_n) if use_rules else hits.head(top_n)
    return result, blocked
