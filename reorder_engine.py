"""
模組:智慧自動叫貨建議引擎

對應規格書:
  4.1 智慧需求預測(基線模型)
  4.3 庫存與安全庫存優化(ABC / XYZ 分類)
  4.2 智慧自動叫貨建議((s, S) 補貨策略)

執行順序就是這三節的順序:先預測賣多少,再算要備多少緩衝,最後決定叫多少。


規格書 4.2 的核心演算法(原文照實作)
------------------------------------
    lead_time_demand = SUM(predicted_qty_p50, days = 1..lead_time_days)
    reorder_point    = lead_time_demand + safety_stock_qty
    target_level_S   = reorder_point + review_period_demand

    IF (current_stock + on_order_qty) < reorder_point:
        order_qty_raw = target_level_S - current_stock - on_order_qty
        order_qty     = CEIL(order_qty_raw / pack_size) * pack_size
        order_qty     = MAX(order_qty, moq)
    ELSE:
        order_qty = 0

    IF ABS(p50 - rolling_avg_4wk) / rolling_avg_4wk > 0.30:
        exception_flag = TRUE


本版對規格書的兩處補充(都標示在輸出欄位裡)
--------------------------------------------
1. **保存期限上限**:規格書 4.2 沒有這一條,但少了它,系統會建議把
   主餐(保存 3 天)備到跟酒水(保存 365 天)一樣的安全庫存水位,
   多出來的部分注定變廚餘。所以叫貨量另外受
   「日均銷量 x 保存期限」的上限約束,並在 exception_reason 註明。

2. **MOQ 造成的過量**:規格書寫 order_qty = MAX(order_qty, moq),
   但如果 MOQ 遠大於實際需求(例如只需要 3 份、MOQ 是 12),
   硬湊到 MOQ 就是製造廚餘。所以當 MOQ 超過保存期上限時會標為例外,
   讓店長決定要不要叫 —— 而不是系統自己決定。


本版限制(誠實記錄,對應工作任務清單)
------------------------------------
  - 預測用「週幾別中位數」的基線模型(規格書 P2-03),不是 P2-04 的
    XGBoost/LightGBM 分位數迴歸。只有 31 天資料,每個週幾僅 4~5 個觀測值,
    梯度提升模型會嚴重過擬合。基線模型在資料量小時反而更穩。
  - 沒有訂位資料 -> 未實作 4.1.3 的 expected_traffic 修正(P1-02 待補)
  - 沒有天氣與行事曆資料 -> 未實作 calendar/weather 特徵(P1-09 待補)
  - 供應商主檔是品類預設值 -> 依規格書 4.2 例外處理標記 needs_data_fill
"""

import numpy as np
import pandas as pd

from ops_data import (banner, build_supplier_master, ensure_output_dir,
                      load_inventory, load_sales_daily, section,
                      service_level_z)

# 規格書 4.2 / 4.3 參數設定
REVIEW_PERIOD_DAYS = 1         # 審查週期:每日一次
EXCEPTION_THRESHOLD = 0.30     # 例外波動門檻 ±30%
SERVICE_LEVEL_BY_ABC = {       # 服務水準依 ABC 分級
    "A": 0.98,                 # 金額貢獻前 80%:斷貨代價最高
    "B": 0.95,
    "C": 0.90,
}


# ------------------------------------------------------------------
# 4.1 需求預測(基線模型)
# ------------------------------------------------------------------
def demand_stats(sales):
    """
    每個品項的需求統計(規格書 4.3 輸入:demand_avg / demand_std)。

    重點:標準差要用**每日**銷量算,不能用週或月的彙總 ——
    安全庫存公式裡的 demand_std 是日需求的標準差,
    用彙總資料算會低估波動,安全庫存就會設太低。
    """
    g = sales.groupby("sku_id")["qty_sold"]
    stats = g.agg(demand_avg="mean", demand_std="std",
                  total_qty="sum", n_days="count").reset_index()

    # rolling_avg_4wk:規格書 4.2 例外判斷的比對基準
    last_day = sales["date"].max()
    win = sales[sales["date"] > last_day - pd.Timedelta(days=28)]
    stats = stats.merge(
        win.groupby("sku_id")["qty_sold"].mean().rename("rolling_avg_4wk"),
        left_on="sku_id", right_index=True, how="left")

    # 變異係數 CV -> XYZ 分類用
    stats["cv"] = stats["demand_std"] / stats["demand_avg"]
    return stats


def forecast_next_days(sales, n_days):
    """
    基線預測:週幾別中位數(P50)與 90 百分位(P90)。

    為什麼要分週幾:餐酒館週五六的來客是平日的兩倍。
    用整體平均去預測週六,會系統性低估;預測週二則會高估。

    回傳 {sku_id: {'p50': 未來 n 天 P50 加總, 'p90': ...}}
    後面 lead_time_demand 直接取 p50 的加總(規格書 4.2 步驟 1)。
    """
    last_day = sales["date"].max()
    future = [last_day + pd.Timedelta(days=i) for i in range(1, n_days + 1)]

    # 週幾 -> 分位數查表
    q = (sales.groupby(["sku_id", "weekday"])["qty_sold"]
              .agg(p50="median", p90=lambda s: s.quantile(0.90)))
    overall = sales.groupby("sku_id")["qty_sold"].agg(p50="median",
                                                      p90=lambda s: s.quantile(0.90))

    out = {}
    for sku in sales["sku_id"].unique():
        p50 = p90 = 0.0
        for d in future:
            key = (sku, d.weekday())
            if key in q.index:
                p50 += float(q.loc[key, "p50"])
                p90 += float(q.loc[key, "p90"])
            else:
                # 該品項在這個週幾沒有紀錄 -> 退回整體分位數
                p50 += float(overall.loc[sku, "p50"])
                p90 += float(overall.loc[sku, "p90"])
        out[sku] = {"p50": p50, "p90": p90}
    return out


# ------------------------------------------------------------------
# 4.3 安全庫存 + ABC / XYZ
# ------------------------------------------------------------------
def classify_abc_xyz(stats, inv):
    """
    ABC:依「用量金額」累積佔比(規格書 4.3)
        cumulative_value_pct <= 80% -> A
                             <= 95% -> B
                             else   -> C

    用量金額 = 銷售份數 x 進貨成本單價。
    注意這是**成本金額**不是營收 —— 庫存管理管的是壓在倉裡的錢,
    所以要用進貨成本,不是售價。

    XYZ:依變異係數 CV = demand_std / demand_avg
        CV < 0.5  -> X(需求穩定,好預測)
        CV <= 1.0 -> Y
        else      -> Z(需求飄忽,要靠安全庫存硬撐)

    ABC x XYZ 交叉才是真正的管理矩陣:
      AX 高金額且穩定  -> 可以壓低庫存,精準叫貨
      AZ 高金額但飄忽  -> 最難管,要最高安全庫存 + 人工盯
      CZ 低金額又飄忽  -> 不值得花力氣,設個固定量就好
    """
    df = stats.merge(inv[["sku_id", "category", "unit_cost", "price",
                          "closing_stock", "on_order_qty"]],
                     on="sku_id", how="left")

    df["usage_value"] = df["total_qty"] * df["unit_cost"]
    df = df.sort_values("usage_value", ascending=False).reset_index(drop=True)
    cum = df["usage_value"].cumsum() / df["usage_value"].sum()
    df["cum_value_pct"] = cum.round(3)
    df["abc_class"] = np.where(cum <= 0.80, "A",
                      np.where(cum <= 0.95, "B", "C"))
    df["xyz_class"] = np.where(df["cv"] < 0.5, "X",
                      np.where(df["cv"] <= 1.0, "Y", "Z"))
    return df


def compute_safety_stock(df, supplier):
    """
    規格書 4.3 安全庫存公式(完整版,含前置時間變異):

        safety_stock = Z x SQRT( LT_avg x demand_std^2
                                 + demand_avg^2 x LT_std^2 )

    兩個項分別在防兩件事:
      第一項 LT_avg x demand_std^2   前置期內「需求波動」帶來的風險
      第二項 demand_avg^2 x LT_std^2 「交期本身不準」帶來的風險

    第二項常被忽略,但在餐飲很關鍵:酒商說 5 天到,實際可能 3 天也可能 8 天。
    交期越不穩(LT_std 越大),要備的安全庫存越多 —— 這跟賣得多不多無關,
    純粹是供應商不可靠的代價。這也是規格書 4.4 要做供應商評分的理由。
    """
    df = df.merge(
        supplier[["sku_id", "lead_time_days_avg", "lead_time_days_std",
                  "moq", "pack_size", "shelf_life_days", "needs_data_fill"]],
        on="sku_id", how="left")

    df["service_level"] = df["abc_class"].map(SERVICE_LEVEL_BY_ABC)
    df["z"] = df["service_level"].apply(service_level_z)

    df["safety_stock_qty"] = (
        df["z"] * np.sqrt(
            df["lead_time_days_avg"] * df["demand_std"] ** 2
            + df["demand_avg"] ** 2 * df["lead_time_days_std"] ** 2
        )
    ).round(1)

    # 審查週期:高金額品項看得比較勤(規格書 4.3 輸出 review_cycle_days)
    df["review_cycle_days"] = df["abc_class"].map({"A": 1, "B": 3, "C": 7})
    return df


# ------------------------------------------------------------------
# 4.2 叫貨建議((s, S) 策略)
# ------------------------------------------------------------------
def build_orders(df, fc):
    """規格書 4.2 演算法逐步實作。"""
    rows = []
    for _, r in df.iterrows():
        lt = max(int(round(r["lead_time_days_avg"])), 1)
        f = fc[r["sku_id"]]

        # 步驟 1:前置期內的預測需求
        # forecast_next_days() 已經按各品項自己的前置期天數加總過 P50,
        # 所以這裡直接取用,不需再乘天數。
        lead_time_demand = f["p50"]

        # 步驟 2~3
        reorder_point = lead_time_demand + r["safety_stock_qty"]
        review_period_demand = r["demand_avg"] * REVIEW_PERIOD_DAYS
        target_level_S = reorder_point + review_period_demand

        available = r["closing_stock"] + r["on_order_qty"]

        reasons = []
        # 步驟 4
        if available < reorder_point:
            raw = target_level_S - available
            pack = max(int(r["pack_size"]), 1)
            qty = float(np.ceil(raw / pack) * pack)
            if qty < r["moq"]:
                qty = float(r["moq"])
                reasons.append(f"未達 MOQ {int(r['moq'])},已補足至最小訂購量")
        else:
            raw, qty = 0.0, 0.0

        # ---- 本版補充 1:保存期限上限 ----
        shelf_cap = r["demand_avg"] * r["shelf_life_days"]
        if qty > 0 and available + qty > shelf_cap:
            capped = max(0.0, shelf_cap - available)
            if capped < qty:
                reasons.append(
                    f"受保存期 {int(r['shelf_life_days'])} 天限制,"
                    f"由 {qty:.0f} 降為 {capped:.0f}(超額部分將成廚餘)")
                qty = capped

        # ---- 本版補充 2:MOQ 超過保存期上限 ----
        if r["moq"] > shelf_cap and available < reorder_point:
            reasons.append(
                f"⚠️ MOQ {int(r['moq'])} 超過保存期內可銷量 {shelf_cap:.0f},"
                f"需與供應商議定拆箱或改期")

        # 步驟 5:需求波動例外
        base = r["rolling_avg_4wk"]
        dev = abs(f["p50"] / max(lt, 1) - base) / base if base else 0.0
        if dev > EXCEPTION_THRESHOLD:
            reasons.append(f"需求波動 {dev:.0%} 超過 {EXCEPTION_THRESHOLD:.0%} 門檻")

        # 注意:needs_data_fill 刻意**不併入 exception_flag**。
        #
        # 第一版把它併進去,結果 12 個品項全部標成「需人工複核」——
        # 全部都是例外等於沒有例外,店長看一次就不會再看。
        # 這和 model_inventory.py 踩過的是同一個坑(提醒疲勞)。
        #
        # 兩者性質不同:
        #   exception_flag  這一張叫貨單本身需要店長判斷(波動、保存期、MOQ)
        #   data_gap_flag   供應商主檔還沒建,是採購的一次性補齊工作(P1-03)
        # 前者是每日的操作提醒,後者是專案代辦事項,應該分開報。

        rows.append({
            "store_id": "S01",
            "sku_id": r["sku_id"],
            "category": r["category"],
            "abc_class": r["abc_class"],
            "xyz_class": r["xyz_class"],
            "service_level": f"{r['service_level']:.0%}",
            "demand_avg": round(r["demand_avg"], 1),
            "demand_std": round(r["demand_std"], 1),
            "cv": round(r["cv"], 2),
            "lead_time_days": lt,
            "predicted_qty_p50_lt": round(f["p50"], 1),
            "predicted_qty_p90_lt": round(f["p90"], 1),
            "safety_stock_qty": r["safety_stock_qty"],
            "reorder_point": round(reorder_point, 1),
            "target_level_S": round(target_level_S, 1),
            "current_stock": r["closing_stock"],
            "on_order_qty": r["on_order_qty"],
            "order_qty_raw": round(raw, 1),
            "recommended_order_qty": int(qty),
            "moq": int(r["moq"]),
            "pack_size": int(r["pack_size"]),
            "shelf_life_days": int(r["shelf_life_days"]),
            "review_cycle_days": int(r["review_cycle_days"]),
            "exception_flag": bool(reasons),
            "exception_reason": " / ".join(reasons),
            "data_gap_flag": bool(r["needs_data_fill"]),
        })

    out = pd.DataFrame(rows)
    return out.sort_values(["recommended_order_qty"], ascending=False)


# ------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------
def main():
    out_dir = ensure_output_dir()
    sales = load_sales_daily()
    inv = load_inventory()
    supplier = build_supplier_master(inv)

    stats = demand_stats(sales)
    df = classify_abc_xyz(stats, inv)
    df = compute_safety_stock(df, supplier)

    # 各品項的前置期不同,所以逐一按自己的前置期天數預測
    fc = {}
    for _, r in df.iterrows():
        lt = max(int(round(r["lead_time_days_avg"])), 1)
        fc[r["sku_id"]] = forecast_next_days(
            sales[sales["sku_id"] == r["sku_id"]], lt)[r["sku_id"]]

    orders = build_orders(df, fc)

    banner("智慧自動叫貨建議(規格書 4.1 / 4.2 / 4.3)")
    n_days = sales["date"].nunique()
    print(f"資料期間 {sales['date'].min():%Y-%m-%d} ~ {sales['date'].max():%Y-%m-%d}"
          f"({n_days} 天)   品項 {len(orders)}")
    print(f"預測模型 週幾別分位數基線(規格書 P2-03);審查週期 "
          f"{REVIEW_PERIOD_DAYS} 天;例外門檻 ±{EXCEPTION_THRESHOLD:.0%}")

    section("ABC x XYZ 管理矩陣")
    mat = pd.crosstab(orders["abc_class"], orders["xyz_class"])
    print(mat.to_string())
    print("\n  AX 高金額且穩定 -> 可壓低庫存、精準叫貨")
    print("  AZ 高金額但飄忽 -> 最難管,要最高安全庫存 + 人工盯")
    print("  CZ 低金額又飄忽 -> 不值得花力氣,設固定量即可")
    if orders["xyz_class"].nunique() == 1:
        cv_max = orders["cv"].max()
        print(f"\n  ⚠️ 12 個品項全部落在 X(CV 最高僅 {cv_max:.2f},門檻 0.5),")
        print("     XYZ 這個維度目前沒有區分力。原因是這份示範資料的日銷量太穩定;")
        print("     真實餐廳受天氣、假日、訂位影響,CV 通常會有品項跨到 Y/Z。")
        print("     -> 換上真實 POS 資料後要重新檢視這個分類的門檻。")

    section("建議叫貨單")
    cols = ["sku_id", "abc_class", "xyz_class", "service_level", "demand_avg",
            "lead_time_days", "safety_stock_qty", "reorder_point",
            "current_stock", "recommended_order_qty"]
    show = orders[cols].copy()
    show.columns = ["品項", "ABC", "XYZ", "服務水準", "日均銷量", "前置期",
                    "安全庫存", "再訂購點", "現有庫存", "建議叫貨"]
    print(show.to_string(index=False))

    section("例外(需店長人工複核)")
    exc = orders[orders["exception_flag"]]
    if exc.empty:
        print("  無")
    else:
        for _, r in exc.iterrows():
            print(f"  {r['sku_id']} — {r['exception_reason']}")

    section("資料缺口(採購/資料工程待補,非每日操作事項)")
    gap = orders[orders["data_gap_flag"]]
    if gap.empty:
        print("  無")
    else:
        print(f"  {len(gap)}/{len(orders)} 品項使用品類預設的供應商參數"
              f"(lead time / MOQ / pack size)")
        print("  -> 對應工作任務清單 P1-03「盤點供應商主檔與報價資料」")
        print("  依規格書 4.2 例外處理:以預設值代入並標記待補齊,不阻擋叫貨建議產生")

    section("摘要")
    need = orders[orders["recommended_order_qty"] > 0]
    print(f"  需叫貨品項      {len(need)} / {len(orders)}")
    print(f"  建議叫貨總份數  {orders['recommended_order_qty'].sum():,} 份")
    cost = (orders["recommended_order_qty"]
            * orders["sku_id"].map(inv.set_index("sku_id")["unit_cost"])).sum()
    print(f"  預估叫貨金額    {cost:,.0f} 元")
    print(f"  需人工複核      {orders['exception_flag'].sum()} 項")
    print(f"  資料待補齊      {orders['data_gap_flag'].sum()} 項")

    orders.to_csv(f"{out_dir}/reorder_recommendation.csv",
                  index=False, encoding="utf-8-sig")
    print("\n已輸出 output/reorder_recommendation.csv")
    return orders


if __name__ == "__main__":
    main()
