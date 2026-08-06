"""
餐酒館智慧點餐 Demo(Streamlit)

模擬外場的 iPad 點餐畫面:
點好餐之後,系統即時推薦「該推哪杯酒」,並說明推薦理由。

執行方式(在 bistro_ai 資料夾下):
    streamlit run app.py

這個介面的設計重點不是好看,而是**讓面試官在 30 秒內看懂價值**:
  左邊點餐 -> 右邊立刻出現推薦 + 推薦理由 + 預估毛利貢獻
  並且可以切換「有/沒有餐飲規則」,直接看出差異
"""

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "3")

import numpy as np
import pandas as pd
import streamlit as st

from model_basket import build_baskets, association_rules
from business_rules import recommend_with_rules

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE_DIR, "data")

st.set_page_config(page_title="餐酒館智慧點餐系統", page_icon="🍷", layout="wide")


# ------------------------------------------------------------------
# 資料載入(快取,避免每次互動都重算關聯規則)
# ------------------------------------------------------------------
@st.cache_data
def load_all():
    visits = pd.read_csv(os.path.join(DATA, "visits.csv"))
    items = pd.read_csv(os.path.join(DATA, "order_items.csv"))
    menu = pd.read_csv(os.path.join(DATA, "menu.csv"))
    rules = association_rules(build_baskets(items), menu)
    return visits, items, menu, rules


@st.cache_resource
def train_upsell(visits):
    """訓練高酒水潛力桌模型(只用入座當下已知的特徵)。"""
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    feats = ["party_size", "arrival_hour", "is_buzz_hour", "is_weekend", "is_repeat"]
    y = (visits["drink_ratio"] >= 0.40).astype(int)
    scaler = StandardScaler().fit(visits[feats])
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(scaler.transform(visits[feats]), y)
    return model, scaler, feats


try:
    visits, items, menu, rules = load_all()
except FileNotFoundError:
    st.error("找不到資料檔。請先在終端機執行:python data_generator.py")
    st.stop()

model, scaler, FEATS = train_upsell(visits)

# ------------------------------------------------------------------
# 標題
# ------------------------------------------------------------------
st.title("🍷 餐酒館智慧點餐與推薦系統")
st.caption(
    "以關聯規則挖掘搭配習慣、以毛利感知排序推薦、再以餐飲專業規則把關 —— "
    "推的不是「客人最可能點的」,而是「客人會接受且最賺錢的」"
)

tab_pos, tab_dash, tab_how = st.tabs(["🧾 點餐推薦", "📊 營運儀表板", "🧠 運作原理"])

# ==================================================================
# Tab 1:點餐推薦
# ==================================================================
with tab_pos:
    left, right = st.columns([1, 1.25])

    with left:
        st.subheader("桌況與點餐")

        c1, c2 = st.columns(2)
        with c1:
            party_size = st.number_input("同桌人數", 1, 10, 2)
            arrival_hour = st.slider("入座時間", 17, 24, 20)
        with c2:
            is_repeat = st.checkbox("常客回訪", value=False)
            is_weekend = st.checkbox("週五 / 週六", value=False)

        st.divider()

        food = menu[menu["type"].isin(["主食", "副食", "甜點"])]
        picked_names = st.multiselect(
            "已點餐點",
            options=list(food["item_name"]),
            default=["舒肥紅酒燉牛肉"] if "舒肥紅酒燉牛肉" in set(food["item_name"]) else [],
        )
        picked_ids = list(food.loc[food["item_name"].isin(picked_names), "item_id"])

        st.divider()
        use_rules = st.toggle(
            "套用餐飲專業規則", value=True,
            help="關閉後只依期望毛利排序,可以看到模型會推出餐飲上錯誤的搭配",
        )

        # ---- 高酒水潛力預測 ----
        row = pd.DataFrame([{
            "party_size": party_size,
            "arrival_hour": arrival_hour,
            "is_buzz_hour": int(arrival_hour >= 21),
            "is_weekend": int(is_weekend),
            "is_repeat": int(is_repeat),
        }])[FEATS]
        prob = float(model.predict_proba(scaler.transform(row))[0, 1])

        st.subheader("這桌的酒水潛力")
        st.progress(prob, text=f"高酒水消費機率 {prob:.0%}")
        if prob >= 0.6:
            st.success("**高潛力桌** — 入座就送酒單,主動介紹當日推薦酒款")
        elif prob >= 0.4:
            st.info("**中等潛力** — 先上餐,主食上桌後再問要不要配一杯")
        else:
            st.warning("**低潛力桌** — 以餐點服務為主,不主動推酒以免造成壓力")

    with right:
        st.subheader("推薦酒款")

        if not picked_ids:
            st.info("請先在左側選擇已點餐點")
        else:
            rec, blocked = recommend_with_rules(
                rules, menu, picked_ids,
                arrival_hour=arrival_hour, top_n=3, use_rules=use_rules,
            )

            if rec.empty:
                st.warning("這些餐點的關聯規則資料不足,無法推薦")
            else:
                for i, (_, r) in enumerate(rec.iterrows(), 1):
                    with st.container(border=True):
                        cc1, cc2, cc3 = st.columns([2.2, 1, 1])
                        cc1.markdown(f"### {i}. {r['consequent']}")
                        cc2.metric("加點成功率", f"{r['confidence']:.0%}")
                        cc3.metric("單杯毛利", f"{r['unit_profit']:.0f} 元")
                        st.caption(
                            f"期望毛利 **{r['expected_profit']:.0f} 元**"
                            f"  ·  lift {r['lift']:.2f}"
                            f"  ·  關聯來源:{r['antecedent']}"
                        )

                total_ep = rec["expected_profit"].sum()
                st.metric("這組推薦的預估毛利貢獻", f"{total_ep:.0f} 元")

            if use_rules and blocked:
                with st.expander(f"⛔ 已擋掉 {len(blocked)} 個餐飲上不合適的搭配"):
                    for b in blocked:
                        st.markdown(f"- **{b['被擋的酒']}**:{b['原因']}")
                    st.caption(
                        "這些組合在統計上會被一起點,但風味衝突。"
                        "模型看得到「一起被點」,看不到「客人喝完的感受」"
                    )

# ==================================================================
# Tab 2:營運儀表板
# ==================================================================
with tab_dash:
    st.subheader("營運現況")

    total_rev = visits["total_revenue"].sum()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("總營收", f"{total_rev/10000:,.0f} 萬")
    m2.metric("平均客單價", f"{visits['spend_per_head'].mean():.0f} 元")
    m3.metric("酒水佔比", f"{visits['drink_revenue'].sum()/total_rev:.1%}",
              help="餐酒館的核心指標,業界目標 30~40%")
    m4.metric("零酒水桌", f"{(visits['n_drink']==0).mean():.1%}",
              help="這些桌是流失的毛利")

    st.divider()
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**各時段桌數(雙峰結構)**")
        st.bar_chart(visits.groupby("arrival_hour").size(), height=260)
        st.caption("19 點晚餐峰 + 21 點微醺峰。一般餐廳只有一個峰,這是餐酒館的結構差異")

    with c2:
        st.markdown("**各時段酒水佔比**")
        st.line_chart(visits.groupby("arrival_hour")["drink_ratio"].mean(), height=260)
        st.caption("越晚酒水佔比越高 —— 這決定了該在哪個時段加強推酒")

    st.divider()
    st.markdown("**停留時間與人均酒水杯數**")
    tmp = visits.copy()
    tmp["dur_bin"] = pd.cut(
        tmp["duration_min"], bins=[0, 60, 90, 120, 150, 300],
        labels=["<60分", "60-90分", "90-120分", "120-150分", ">150分"],
    )
    per = tmp.groupby("dur_bin", observed=True).apply(
        lambda g: (g["n_drink"] / g["party_size"]).mean(), include_groups=False
    )
    st.bar_chart(per, height=240)
    st.caption(
        "停留越久喝越多 —— 所以餐酒館不該追求翻桌率。"
        "這跟一般餐廳的營運邏輯完全相反,是這門生意最關鍵的認知"
    )

    st.divider()
    st.markdown("**品項毛利貢獻 Top 10**")
    prof = (items.groupby("item_name")["profit"].sum()
                 .sort_values(ascending=False).head(10))
    st.bar_chart(prof, height=280)

# ==================================================================
# Tab 3:運作原理
# ==================================================================
with tab_how:
    st.subheader("三階段推薦流程")

    st.markdown(
        """
| 階段 | 做什麼 | 用到的方法 |
|---|---|---|
| **① 召回** | 找出所有和已點餐點有關聯的酒款 | Apriori 關聯規則(support / confidence / lift) |
| **② 排序** | 依「期望毛利」排序,而非「加點成功率」 | 期望毛利 = confidence × 售價 × 毛利率 |
| **③ 重排** | 用餐飲專業規則把關 | 搭配禁忌 / 時段調整 / 推薦多樣性 |
"""
    )

    st.divider()
    st.markdown("#### 為什麼要用「期望毛利」而不是「加點成功率」")
    st.markdown(
        """
一般推薦系統推的是「最可能被一起點的品項」,但那通常是最便宜、毛利最低的。
推薦成功了,卻沒賺到錢。

| | A 酒 | B 酒 |
|---|---|---|
| 加點成功率 | 60% | 40% |
| 單杯毛利 | 100 元 | 266 元 |
| **期望毛利** | **60 元** | **106 元** |

結論是該推 B,即使它「比較難成功」。
這個轉換就是餐飲營運的思維 —— 目標是每桌毛利,不是推薦命中率。
        """
    )

    st.divider()
    st.markdown("#### 為什麼需要第三階段(餐飲規則)")
    st.markdown(
        """
實測時模型算出「香煎鮭魚佐檸檬」該配「Negroni」,因為 Negroni 毛利最高、
統計上也常被一起點。

但這在餐飲上是錯的:Negroni 的金巴利苦味會蓋掉鮭魚的細緻風味,
苦韻還會把海鮮的腥味帶出來。這杯酒推出去,客人喝一口就放下,下次不會再來。

**模型看得到「一起被點」,看不到「客人喝完的感受」。**

第三階段就是把餐飲專業知識寫成規則替模型把關 ——
也是這個專題最難被取代的部分。
        """
    )

    st.divider()
    st.markdown("#### 資料說明")
    st.info(
        "本專題使用模擬資料(150 天、約 1,800 桌)。真實餐廳 POS 資料屬營業機密,"
        "公開資料集中沒有餐酒館的完整消費紀錄。\n\n"
        "模擬資料的行為規則全部依照餐酒館實際運作設定:雙峰到店時間、"
        "停留時間與酒水杯數的正相關、風味搭配傾向、常客與新客的消費差異等。"
        "資料是模擬的,但**營運邏輯是真的**。"
    )
