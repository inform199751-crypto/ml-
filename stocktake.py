"""
模組:盤點與耗損異常偵測

對應規格書 4.5「異常偵測與耗損管理」。

盤點在做什麼
------------
月底清點實際還剩多少,和「帳上應該剩多少」比對:

    理論庫存 = 期初庫存 + 本月叫貨 - 本月銷售
    實際盤點 = 人去數出來的數字
    差異     = 理論 - 實際

差異的來源可能是:耗損(切壞、掉地上、過期)、試菜、員工餐、
未登記的贈品、銷售登記錯誤、盤點數錯,或者偷竊。
盤點的目的不是抓人,是**找出帳與現實脫節的地方**。


⚠️ 這個模組修正了原始報表的一個關鍵錯誤
----------------------------------------
`河岸餐酒館_營運分析.xlsx` 的「差異率」是這樣算的:

    差異率 = 差異 ÷ 理論庫存        <- 分母是「期末還剩多少」

這個分母選錯了。主廚牛排月底理論庫存只剩 7 份,差異 4 份,
算出來就是 4/7 = **57.14%**,看起來像有人在偷牛排。

但規格書 4.5 定義的差異率是**用量基準**:

    實際用量   = 期初 + 叫貨 - 實際盤點
    理論用量   = 銷售量
    差異率     = (實際用量 - 理論用量) ÷ 理論用量

主廚牛排:(33 + 194 - 3) ÷ 220 - 1 = **1.8%** —— 完全正常的耗損。

為什麼分母不同會差這麼多:期末庫存是「剩下的零頭」,
它的大小取決於這個月最後幾天賣得多不多,和耗損本身沒有關係。
賣得越好、期末剩越少,分母越小,同樣的 4 份差異就被放大成越誇張的百分比。

**如果照原表那個 57% 去查,會冤枉現場員工。** 這是分母選錯造成的假警報,
而不是真的有異常 —— 這也是這個模組存在的理由。
"""

import numpy as np
import pandas as pd

from ops_data import (banner, build_supplier_master, ensure_output_dir,
                      load_inventory, load_sales_daily, section)

# 規格書 4.5 參數設定
VARIANCE_THRESHOLD = 0.15      # 差異率告警門檻 ±15%
IF_CONTAMINATION = 0.05        # Isolation Forest 預期異常比例
IF_MIN_ROWS = 60               # 低於這個列數就不跑無監督模型(理由見下方)


# ------------------------------------------------------------------
# 步驟一:用量基準的差異率(規格書 4.5)
# ------------------------------------------------------------------
def compute_variance(inv):
    """
    計算實際用量、理論用量與差異率。

    目前沒有 BOM(品項配方表,工作任務清單 P1-05 待補),
    所以理論用量直接用「銷售份數」。
    BOM 建好之後,理論用量要改成:
        theoretical_usage = SUM(recipe_qty_per_unit x qty_sold)
    也就是把「賣了 220 份牛排」換算成「該用掉 220 x 0.25kg = 55kg 牛肉」,
    才能抓到原物料層級的耗損。這是目前這版最大的限制。
    """
    df = inv.copy()

    # 實際用量:進了多少、剩多少,中間消失的就是實際用掉的
    df["actual_usage"] = (df["opening_stock"] + df["received_qty"]
                          - df["closing_stock"])
    # 理論用量:帳上賣掉的份數(BOM 補齊後改為原物料換算)
    df["theoretical_usage"] = df["qty_sold"]

    df["variance_qty"] = df["actual_usage"] - df["theoretical_usage"]
    df["variance_ratio"] = df["variance_qty"] / df["theoretical_usage"]

    # 帳面庫存差異(用來對帳,不用來判斷耗損)
    df["stock_variance_qty"] = df["theoretical_stock"] - df["closing_stock"]

    # 耗損金額 = 差異份數 x 進貨成本單價
    df["waste_cost"] = df["variance_qty"] * df["unit_cost"]
    return df


def classify_alert(df):
    """
    規則型告警(規格書 4.5)。

    三種情況要分開處理,因為處置方式完全不同:

      差異率 > +15%   實際用量大於帳上該用的量
                      -> 真的有東西消失了:耗損、試菜未登記、或盤損
      差異率 < -15%   實際用量小於帳上該用的量(庫存比帳上多)
                      -> 這**不是耗損**,是資料錯誤:
                         叫貨登記漏了、銷售重複登記、或盤點數錯。
                         負差異一定要查,但要查的是「帳」不是「人」
      其他            正常範圍
    """
    def level(v):
        if not np.isfinite(v):
            return "低", "無銷售紀錄,無法判斷"
        if v > VARIANCE_THRESHOLD:
            return "中", f"實際用量超出理論用量 {v:.1%},超過 ±15% 門檻"
        if v < -VARIANCE_THRESHOLD:
            return "中", (f"實際用量低於理論用量 {abs(v):.1%} —— "
                          f"庫存比帳上多,屬**資料登記問題**而非耗損")
        return "低", "差異率在正常範圍內"

    lv = df["variance_ratio"].apply(level)
    df["alert_level"] = [x[0] for x in lv]
    df["alert_reason"] = [x[1] for x in lv]
    return df


# ------------------------------------------------------------------
# 步驟二:無監督異常偵測(規格書 4.5 Isolation Forest)
# ------------------------------------------------------------------
def detect_anomaly(df):
    """
    規格書要求用 IsolationForest(contamination=0.05) 補足規則型偵測。

    但這一版**刻意不跑模型**,原因要講清楚:

    現在的資料是 12 個品項 x 1 個月 = 12 列。
    contamination=0.05 表示預期 5% 是異常 -> 12 x 0.05 = 0.6 個異常。
    在 12 列資料上跑 IsolationForest,分出來的「異常」和隨機挑一個沒有差別;
    而且規格書要求異常偵測 Precision >= 60%,12 列不可能驗證這個指標。

    **模型不是想用就能用。** 資料量不夠的時候,規則型偵測(±15% 門檻)
    比無監督模型可靠得多,而且店長看得懂為什麼被標記 —— 這在導入初期更重要,
    因為沒人會信一個說不出理由的黑盒子。

    什麼時候才該開啟:
      規格書的 Fact_Inventory 粒度是 store_id x sku_id x **date**(每日)。
      等 ETL(P1-08)建好、有每日庫存異動,資料量會是
      12 品項 x 30 天 x N 家店 = 數百到數千列,那時候模型才有意義。
      門檻設在 IF_MIN_ROWS = 60 列。
    """
    n = len(df)
    if n < IF_MIN_ROWS:
        df["anomaly_score"] = np.nan
        df["anomaly_flag"] = False
        return df, (f"資料僅 {n} 列,低於 {IF_MIN_ROWS} 列門檻 -> "
                    f"跳過 IsolationForest,以規則型偵測為準")

    from sklearn.ensemble import IsolationForest

    feats = pd.DataFrame({
        "variance_ratio": df["variance_ratio"].fillna(0),
        "waste_qty_ratio": (df["variance_qty"].abs()
                            / df["theoretical_usage"].replace(0, np.nan)).fillna(0),
        "sku_category_encoded": pd.factorize(df["category"])[0],
    })
    model = IsolationForest(contamination=IF_CONTAMINATION, random_state=42)
    model.fit(feats)
    df["anomaly_score"] = model.decision_function(feats).round(4)
    df["anomaly_flag"] = model.predict(feats) == -1
    df.loc[df["anomaly_flag"], "alert_level"] = "高"
    return df, f"IsolationForest 已執行({n} 列,contamination={IF_CONTAMINATION})"


# ------------------------------------------------------------------
# 步驟三:斷貨風險(盤點的另一半)
# ------------------------------------------------------------------
def stockout_risk(df, sales):
    """
    盤點不只要看「少了多少」,還要看「剩下的夠不夠賣」。

    這是原始報表完全沒有的一段,但它才是主廚牛排真正的問題:
    耗損 1.8% 是正常的,**庫存只剩 3 份才是要命的** ——
    日均銷量 7.1 份,連半天都撐不到。
    """
    daily = (sales.groupby("sku_id")["qty_sold"].sum()
             / sales["date"].nunique()).rename("avg_daily")
    out = df.merge(daily, left_on="sku_id", right_index=True, how="left")
    out["days_of_cover"] = (out["closing_stock"] / out["avg_daily"]).round(2)
    out["stockout_risk"] = np.where(
        out["days_of_cover"] < 1.0, "🔴 不足 1 日",
        np.where(out["days_of_cover"] < 3.0, "🟡 不足 3 日", "🟢 充足"))
    return out


# ------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------
def main():
    out_dir = ensure_output_dir()
    inv = load_inventory()
    sales = load_sales_daily()
    _ = build_supplier_master(inv)

    df = compute_variance(inv)
    df = classify_alert(df)
    df, if_note = detect_anomaly(df)
    df = stockout_risk(df, sales)

    banner("盤點與耗損異常偵測(規格書 4.5)")
    print(f"分析期間 2026-07   品項數 {len(df)}   "
          f"差異率門檻 ±{VARIANCE_THRESHOLD:.0%}")
    print(f"無監督偵測:{if_note}")

    # ---- 差異率:新舊定義對照 ----
    section("差異率:原始報表 vs 規格書 4.5 定義")
    cmp = df[["sku_id", "qty_sold", "closing_stock", "theoretical_stock",
              "variance_qty", "variance_ratio_legacy", "variance_ratio"]].copy()
    cmp.columns = ["品項", "銷售量", "實際盤點", "理論庫存",
                   "差異份數", "原表差異率", "正確差異率"]
    cmp["原表差異率"] = cmp["原表差異率"].map("{:.1%}".format)
    cmp["正確差異率"] = cmp["正確差異率"].map("{:.1%}".format)
    print(cmp.sort_values("差異份數", ascending=False).to_string(index=False))
    print("\n  原表差異率 = 差異 ÷ 理論庫存(期末剩餘量)-> 期末剩越少,百分比被放大越誇張")
    print("  正確差異率 = (實際用量 - 理論用量) ÷ 理論用量  <- 規格書 4.5 定義")

    # ---- 告警 ----
    section("耗損告警")
    alerts = df[df["alert_level"].isin(["中", "高"])]
    if alerts.empty:
        print(f"  無。全部品項差異率都在 ±{VARIANCE_THRESHOLD:.0%} 內")
        print(f"  實際區間 {df['variance_ratio'].min():+.1%} ~ "
              f"{df['variance_ratio'].max():+.1%},屬餐飲業正常耗損水準")
        print("\n  ⚠️ 對照:若沿用原表的差異率定義,會有 "
              f"{(df['variance_ratio_legacy'].abs() > VARIANCE_THRESHOLD).sum()} "
              "個品項被誤標為異常")
    else:
        cols = ["sku_id", "category", "variance_ratio", "waste_cost",
                "alert_level", "alert_reason"]
        print(alerts[cols].to_string(index=False))

    # ---- 斷貨風險 ----
    section("斷貨風險(盤點的另一半)")
    risk = df[df["stockout_risk"] != "🟢 充足"].sort_values("days_of_cover")
    if risk.empty:
        print("  無")
    else:
        cols = ["sku_id", "category", "closing_stock", "avg_daily",
                "days_of_cover", "stockout_risk"]
        r = risk[cols].copy()
        r.columns = ["品項", "分類", "實際盤點", "日均銷量", "可撐天數", "風險"]
        print(r.to_string(index=False))
        print("\n  這一段是原始報表沒有的。主廚牛排的問題不是耗損(1.8% 正常),")
        print("  是庫存只剩 3 份、日均賣 7.1 份 —— 連半天都撐不到。")

    # ---- 摘要 ----
    section("摘要")
    total_waste = df.loc[df["variance_qty"] > 0, "waste_cost"].sum()
    print(f"  正向耗損成本合計   {total_waste:,.0f} 元")
    print(f"  耗損率(用量基準)  {df['variance_qty'].clip(lower=0).sum() / df['theoretical_usage'].sum():.2%}")
    print(f"  規則型告警         {(df['alert_level'] == '中').sum()} 項")
    print(f"  斷貨風險           {(df['stockout_risk'] != '🟢 充足').sum()} 項")
    neg = df[df["variance_qty"] < 0]
    if not neg.empty:
        print(f"  ⚠️ 負差異(帳務問題)  {len(neg)} 項:"
              f"{', '.join(neg['sku_id'])} —— 庫存比帳上多,要查登記不是查人")

    path = f"{out_dir}/stocktake_variance.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n已輸出 output/stocktake_variance.csv")
    return df


if __name__ == "__main__":
    main()
