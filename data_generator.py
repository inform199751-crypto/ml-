"""
餐酒館 POS 資料產生器

為什麼要自己產生資料?
  真實餐廳 POS 資料屬於營業機密,公開資料集裡幾乎找不到餐酒館的完整消費紀錄。
  與其硬套一個不相關的公開資料集,不如自己建一個「行為邏輯正確」的模擬資料。

這個檔案是整個專題最關鍵的差異化來源。
一般人寫模擬資料會用均勻隨機 (uniform random),產出的資料沒有營運意義;
這裡每一條規則都是餐酒館的實際運作邏輯:

  1. 週五、六來客量是平日的兩倍以上,週一公休
  2. 到店時間是「雙峰」—— 19 點晚餐峰 + 21 點微醺峰
     (一般餐廳只有一個晚餐峰,這是餐酒館的結構性差異)
  3. 停留時間越長,酒水杯數越多(這是餐酒館的獲利機制)
  4. 21 點後入座的客人,吃得少、喝得多(第二輪客人是來喝的)
  5. 風味搭配有傾向性:點紅肉的客人偏紅酒,點海鮮的偏白酒
  6. 常客的酒水消費明顯高於新客(他們知道要點什麼)

產出兩張表:
  data/visits.csv       一桌一列(彙總),用於加點預測與客群分群
  data/order_items.csv  一個品項一列(明細),用於購物籃關聯分析
"""

import os
import sys
import numpy as np
import pandas as pd

from config import (
    MENU, MENU_COLUMNS, BUSINESS, WEEKDAY_FACTOR,
    ARRIVAL_WEIGHTS, PAIRING_AFFINITY, SIM, DATA_DIR,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_menu_df():
    """回傳菜單 DataFrame。"""
    return pd.DataFrame(MENU, columns=MENU_COLUMNS)


def _pick_party_size(rng, arrival_hour):
    """
    決定同桌人數。
    餐酒館以 2 人為主(約會、朋友小酌),這跟家庭餐廳很不一樣。
    22 點後幾乎不會有大桌 —— 那個時間帶小孩或長輩來的客人已經離場。
    """
    if arrival_hour >= 22:
        return int(rng.choice([1, 2, 3, 4], p=[0.12, 0.58, 0.20, 0.10]))
    return int(rng.choice([1, 2, 3, 4, 5, 6], p=[0.06, 0.46, 0.18, 0.18, 0.07, 0.05]))


def _pick_duration(rng, party_size, arrival_hour):
    """
    停留時間(分鐘)。
    餐酒館是「慢桌」生意 —— 翻桌率不是它的 KPI,客單價才是。
    人越多聊越久;21 點後入座的客人停留也偏長(沒有時間壓力)。
    """
    base = 78 + 11 * (party_size - 2)
    if arrival_hour >= BUSINESS["buzz_hour"]:
        base += 22
    dur = rng.normal(base, 26)
    return int(np.clip(dur, 35, 240))


def _pick_food(rng, menu, party_size, arrival_hour):
    """
    點餐:主食 + 副食 + 甜點。

    餐酒館的點法是「分享式」—— 2 個人不會點 2 份主食,
    而是 1 份主食配 2~3 份 Tapas 一起分。
    21 點後入座的客人主食點得更少(他們是來喝的,不是來吃飯的)。
    """
    mains = menu[menu["type"] == "主食"]
    sides = menu[menu["type"] == "副食"]
    desserts = menu[menu["type"] == "甜點"]

    # 主食:約人數的 0.55~0.95 倍(分享),微醺時段再打折
    main_rate = rng.uniform(0.55, 0.95)
    if arrival_hour >= BUSINESS["buzz_hour"]:
        main_rate *= 0.55
    n_main = max(0 if arrival_hour >= 22 else 1, int(round(party_size * main_rate)))

    # 副食:Tapas 是加點主力,人越多點越多
    n_side = int(np.clip(rng.poisson(1.1 + 0.45 * party_size), 0, 7))

    # 甜點:約三成的桌會點,大桌更可能
    n_dessert = 1 if rng.random() < (0.16 + 0.05 * party_size) else 0

    picked = []
    if n_main > 0:
        picked += list(rng.choice(mains["item_id"], size=n_main, replace=True))
    if n_side > 0:
        picked += list(rng.choice(sides["item_id"], size=n_side, replace=True))
    if n_dessert > 0:
        picked += list(rng.choice(desserts["item_id"], size=n_dessert, replace=True))
    return picked


def _pick_drinks(rng, menu, food_ids, party_size, duration, arrival_hour, is_repeat):
    """
    點酒 —— 這是整個模擬最重要的部分,因為酒水是餐酒館的獲利來源。

    杯數由四個因素決定:
      1. 停留時間:每多待 45 分鐘,大約多一輪酒
      2. 微醺時段:21 點後入座的客人,人均杯數明顯上升
      3. 人數:人多互相帶動
      4. 常客:知道自己要喝什麼,點得比新客多

    酒的種類則由「已點餐點的風味標籤」決定(PAIRING_AFFINITY),
    這樣產出的資料才會有真實的搭配關聯 —— 也才有東西讓 Apriori 挖出來。
    """
    menu_idx = menu.set_index("item_id")
    drinks = menu[menu["type"] == "酒水"]

    # ---- 決定杯數 ----
    rate = 0.75                              # 每人基礎杯數
    rate += 0.55 * (duration / 60 - 1.3)     # 停留時間影響
    if arrival_hour >= BUSINESS["buzz_hour"]:
        rate += 0.55                         # 微醺時段加成
    if is_repeat:
        rate += 0.25                         # 常客加成
    rate = max(0.15, rate)

    n_drink = int(np.clip(rng.poisson(rate * party_size), 0, 14))
    if n_drink == 0:
        return []

    # ---- 決定種類:依照已點餐點的風味標籤加權 ----
    food_tags = [menu_idx.loc[i, "pairing_tag"] for i in food_ids] or ["Universal"]

    # 把每道菜的搭配傾向加總,得到這一桌的整體酒類偏好
    tag_score = {}
    for ft in food_tags:
        for wine_tag, w in PAIRING_AFFINITY.get(ft, {}).items():
            tag_score[wine_tag] = tag_score.get(wine_tag, 0.0) + w

    tags = list(tag_score.keys())
    probs = np.array([tag_score[t] for t in tags], dtype=float)
    probs = probs / probs.sum()

    picked = []
    for _ in range(n_drink):
        want_tag = rng.choice(tags, p=probs)
        pool = drinks[drinks["pairing_tag"] == want_tag]
        if pool.empty:                       # 該風味沒有對應酒款就退回全酒單
            pool = drinks
        picked.append(str(rng.choice(pool["item_id"])))
    return picked


def generate(days=None, seed=None, verbose=True):
    """產生模擬資料,回傳 (visits_df, items_df, menu_df)。"""
    days = days or SIM["days"]
    seed = SIM["seed"] if seed is None else seed
    rng = np.random.default_rng(seed)

    menu = get_menu_df()
    menu_idx = menu.set_index("item_id")

    # 建立客群池:兩成是常客,他們會重複回訪
    n_customers = int(days * 9)
    regular_ids = [f"C{i:05d}" for i in range(int(n_customers * 0.20))]
    new_pool = [f"C{i:05d}" for i in range(len(regular_ids), n_customers)]

    hours = list(ARRIVAL_WEIGHTS.keys())
    hour_p = np.array([ARRIVAL_WEIGHTS[h] for h in hours], dtype=float)
    hour_p /= hour_p.sum()

    visits, items = [], []
    visit_no = 0
    dates = pd.date_range(SIM["start_date"], periods=days, freq="D")

    for d in dates:
        wd = d.weekday()
        factor = WEEKDAY_FACTOR[wd]
        if factor <= 0:                       # 週一公休
            continue

        # 當日桌數:以座位數與週幾強度推估,再加上隨機波動
        n_tables = int(np.clip(rng.normal(15 * factor, 3.0), 3, 60))

        for _ in range(n_tables):
            visit_no += 1
            arrival_hour = int(rng.choice(hours, p=hour_p))
            party_size = _pick_party_size(rng, arrival_hour)
            duration = _pick_duration(rng, party_size, arrival_hour)

            # 三成的桌是常客回訪
            is_repeat = rng.random() < 0.30
            customer_id = str(rng.choice(regular_ids if is_repeat else new_pool))

            food_ids = _pick_food(rng, menu, party_size, arrival_hour)
            drink_ids = _pick_drinks(
                rng, menu, food_ids, party_size, duration, arrival_hour, is_repeat
            )
            all_ids = food_ids + drink_ids
            if not all_ids:
                continue

            visit_id = f"V{visit_no:06d}"

            food_rev = drink_rev = food_profit = drink_profit = 0.0
            for iid in all_ids:
                row = menu_idx.loc[iid]
                price = float(row["price"])
                profit = price * float(row["profit_margin"])
                is_drink = row["type"] == "酒水"
                if is_drink:
                    drink_rev += price
                    drink_profit += profit
                else:
                    food_rev += price
                    food_profit += profit

                items.append({
                    "visit_id": visit_id,
                    "date": d.date(),
                    "item_id": iid,
                    "item_name": row["item_name"],
                    "type": row["type"],
                    "pairing_tag": row["pairing_tag"],
                    "price": price,
                    "profit": round(profit, 1),
                })

            total_rev = food_rev + drink_rev
            visits.append({
                "visit_id": visit_id,
                "customer_id": customer_id,
                "date": d.date(),
                "weekday": wd,
                "is_weekend": int(wd in (4, 5)),
                "arrival_hour": arrival_hour,
                "is_buzz_hour": int(arrival_hour >= BUSINESS["buzz_hour"]),
                "party_size": party_size,
                "duration_min": duration,
                "is_repeat": int(is_repeat),
                "n_items": len(all_ids),
                "n_food": len(food_ids),
                "n_drink": len(drink_ids),
                "food_revenue": round(food_rev, 1),
                "drink_revenue": round(drink_rev, 1),
                "total_revenue": round(total_rev, 1),
                "total_profit": round(food_profit + drink_profit, 1),
                # 酒水佔比:餐酒館最核心的營運指標
                "drink_ratio": round(drink_rev / total_rev, 4) if total_rev else 0.0,
                "spend_per_head": round(total_rev / party_size, 1),
            })

    visits_df = pd.DataFrame(visits)
    items_df = pd.DataFrame(items)

    if verbose:
        print(f"產生完成:{len(visits_df):,} 桌 / {len(items_df):,} 筆品項明細")
        print(f"期間:{dates[0].date()} ~ {dates[-1].date()}")
        print(f"平均客單價:{visits_df['spend_per_head'].mean():.0f} 元/人")
        print(f"平均酒水佔比:{visits_df['drink_ratio'].mean():.1%}")

    return visits_df, items_df, menu


def main():
    out_dir = os.path.join(BASE_DIR, DATA_DIR)
    os.makedirs(out_dir, exist_ok=True)

    visits, items, menu = generate()

    visits.to_csv(os.path.join(out_dir, "visits.csv"), index=False, encoding="utf-8-sig")
    items.to_csv(os.path.join(out_dir, "order_items.csv"), index=False, encoding="utf-8-sig")
    menu.to_csv(os.path.join(out_dir, "menu.csv"), index=False, encoding="utf-8-sig")

    print(f"\n已輸出到 {out_dir}/")
    print("  visits.csv       一桌一列(用於加點預測、客群分群)")
    print("  order_items.csv  品項明細(用於購物籃關聯分析)")
    print("  menu.csv         菜單")


if __name__ == "__main__":
    main()
