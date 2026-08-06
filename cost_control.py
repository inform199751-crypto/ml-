"""
模組:成本控管

對應規格書:
  4.4 供應商評估與採購優化(加權評分 + 跨店聯合採購 MILP)
  企劃書第七節 ROI(食材成本率、耗損率的效益基準)

成本控管在回答四個問題
----------------------
  Q1 錢花在哪裡?        成本結構 vs 業界標竿
  Q2 食材成本對得起帳嗎? 理論食材成本 vs 實際叫貨金額
  Q3 哪些品項在吃毛利?   品項層級的成本貢獻與毛利率
  Q4 供應商該換嗎?      規格書 4.4 加權評分

Q2 是最容易被忽略、但最能抓到問題的一題
--------------------------------------
一般報表只看「食材成本率 = 叫貨金額 ÷ 營業額」。
但叫貨金額包含了「進來還沒賣掉的庫存」,所以這個比率會被叫貨節奏干擾:
月底多叫一批,成本率立刻變高,但那不是真的變貴,只是貨還在倉裡。

正確的做法是把三者分開:

    理論食材成本 = SUM(銷售份數 x 進貨成本單價)   <- 真正賣掉的東西值多少
    實際叫貨金額 = SUM(叫貨份數 x 進貨成本單價)   <- 這個月付了多少錢
    差額         = 庫存變動 + 耗損

差額如果是正的,錢變成了庫存(壓資金但沒虧);
如果理論成本 > 實際叫貨金額,代表在吃上個月的庫存。
兩者都不是「成本失控」,但一般報表會把它們誤讀成成本失控。
"""

import numpy as np
import pandas as pd

from ops_data import (FOOD_COST_BENCHMARK, banner, build_supplier_master,
                      ensure_output_dir, load_cost_summary, load_inventory,
                      load_sales_daily, section)

# 規格書 4.4 供應商評分權重(初始建議值,需與採購團隊確認)
SCORE_WEIGHTS = {
    "on_time_rate": 0.35,
    "quality": 0.30,            # 1 - defect_rate
    "price_competitiveness": 0.25,
    "payment_term_score": 0.10,
}


# ------------------------------------------------------------------
# Q1 成本結構 vs 業界標竿
# ------------------------------------------------------------------
def cost_structure(summary, revenue):
    """把成本支出分析的絕對金額,換算成佔營業額的比例並對標。"""
    items = [
        ("食材成本", summary.get("食材成本(叫貨,元)", 0), FOOD_COST_BENCHMARK),
        ("人事費用", summary.get("人事費用(元)", 0), (0.25, 0.32)),
        ("房租", summary.get("房租(元)", 0), (0.08, 0.15)),
        ("水電雜支", summary.get("水電雜支(元)", 0), (0.03, 0.06)),
        ("行銷雜支", summary.get("行銷雜支(元)", 0), (0.01, 0.03)),
    ]
    rows = []
    for name, amt, (lo, hi) in items:
        pct = amt / revenue if revenue else 0
        if pct > hi:
            verdict, gap = "🔴 偏高", (pct - hi) * revenue
        elif pct < lo:
            verdict, gap = "🟢 低於標竿", 0.0
        else:
            verdict, gap = "🟡 區間內", 0.0
        rows.append({
            "項目": name, "金額": amt, "佔營業額": pct,
            "業界標竿": f"{lo:.0%}~{hi:.0%}", "判定": verdict,
            "超標金額": gap,
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Q2 理論食材成本 vs 實際叫貨金額
# ------------------------------------------------------------------
def food_cost_reconciliation(inv, revenue, actual_purchase):
    """
    三個數字拆開對帳。這一段是這個模組最有價值的地方。
    """
    inv = inv.copy()
    inv["theoretical_cost"] = inv["qty_sold"] * inv["unit_cost"]
    inv["purchase_cost"] = inv["received_qty"] * inv["unit_cost"]
    inv["stock_change_qty"] = inv["closing_stock"] - inv["opening_stock"]
    inv["stock_change_cost"] = inv["stock_change_qty"] * inv["unit_cost"]

    # 耗損 = 實際用量 - 理論用量(與 stocktake.py 同一定義)
    inv["waste_qty"] = (inv["opening_stock"] + inv["received_qty"]
                        - inv["closing_stock"] - inv["qty_sold"])
    inv["waste_cost"] = inv["waste_qty"] * inv["unit_cost"]

    theo = inv["theoretical_cost"].sum()
    purch = inv["purchase_cost"].sum()
    stock_chg = inv["stock_change_cost"].sum()
    waste = inv.loc[inv["waste_qty"] > 0, "waste_cost"].sum()

    return inv, {
        "營業額": revenue,
        "理論食材成本(賣掉的)": theo,
        "實際叫貨金額(付出去的)": purch,
        "報表食材成本": actual_purchase,
        "庫存變動": stock_chg,
        "耗損成本": waste,
        "理論成本率": theo / revenue if revenue else 0,
        "叫貨成本率": purch / revenue if revenue else 0,
    }


# ------------------------------------------------------------------
# Q3 品項層級毛利
# ------------------------------------------------------------------
def item_margin(inv):
    """
    品項毛利分析。

    注意要同時看**毛利率**和**毛利額**:
    毛利率高但賣得少的品項,對整體毛利沒有貢獻;
    毛利率低但賣爆的品項(帶客款),反而是撐營收的主力 —— 不能因為率低就砍。
    """
    df = inv.copy()
    df["unit_margin"] = df["price"] - df["unit_cost"]
    df["margin_rate"] = df["unit_margin"] / df["price"]
    df["revenue"] = df["qty_sold"] * df["price"]
    df["total_margin"] = df["qty_sold"] * df["unit_margin"]
    df["cost_share"] = (df["qty_sold"] * df["unit_cost"])
    df["cost_share"] = df["cost_share"] / df["cost_share"].sum()
    df["margin_share"] = df["total_margin"] / df["total_margin"].sum()

    # 標記:成本佔比 > 毛利佔比 -> 這個品項吃掉的成本比它貢獻的毛利多
    df["efficiency"] = np.where(
        df["cost_share"] > df["margin_share"], "⚠️ 成本佔比 > 毛利佔比", "")
    return df.sort_values("total_margin", ascending=False)


# ------------------------------------------------------------------
# Q4 供應商評分(規格書 4.4)
# ------------------------------------------------------------------
def score_suppliers(supplier):
    """
    規格書 4.4 加權評分:

        score = 100 * ( 0.35 * on_time_rate
                      + 0.30 * (1 - defect_rate)
                      + 0.25 * price_competitiveness
                      + 0.10 * payment_term_score )

    ⚠️ 這裡的四個輸入指標**目前完全沒有資料**:
       on_time_rate 要驗收紀錄、defect_rate 要退貨紀錄,
       price_competitiveness 要至少兩家供應商的比價。
       這些都在工作任務清單 P1-03,尚未盤點。

    所以本函式只實作「評分邏輯」與「輸入契約」,
    並用中性值(0.5 / 業界平均)代入示範。**輸出的分數不可用於決策** ——
    刻意標記在 data_status 欄位,避免下游誤用。

    為什麼還是要先寫:導入專案裡先把介面與公式定下來,
    採購補資料時才知道要補成什麼格式。這比等資料到了再想公式快得多。
    """
    rows = []
    for _, r in supplier.drop_duplicates("supplier_id").iterrows():
        # 帳期分數:帳期越長對現金流越好,45 天以上給滿分
        payment_term_score = min(r["payment_term_days"] / 45.0, 1.0)

        metrics = {
            "on_time_rate": 0.50,             # 佔位值,無驗收紀錄
            "quality": 0.50,                  # 佔位值,無退貨紀錄
            "price_competitiveness": 0.50,    # 佔位值,無比價資料
            "payment_term_score": payment_term_score,
        }
        score = 100 * sum(SCORE_WEIGHTS[k] * v for k, v in metrics.items())
        rows.append({
            "supplier_id": r["supplier_id"],
            "category": r["category"],
            "lead_time_avg": r["lead_time_days_avg"],
            "lead_time_std": r["lead_time_days_std"],
            "payment_term_days": int(r["payment_term_days"]),
            "payment_term_score": round(payment_term_score, 2),
            "score": round(score, 1),
            "data_status": "❌ 僅帳期為真實值,其餘為佔位值 -> 分數不可用於決策",
        })
    out = pd.DataFrame(rows).sort_values("score", ascending=False)
    out["rank"] = range(1, len(out) + 1)
    return out


# ------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------
def main():
    out_dir = ensure_output_dir()
    inv = load_inventory()
    sales = load_sales_daily()
    summary = load_cost_summary()
    supplier = build_supplier_master(inv)

    revenue = summary.get("營業額(元)", sales["amount"].sum())
    actual_purchase = summary.get("食材成本(叫貨,元)", 0)

    banner("成本控管(規格書 4.4 / 企劃書第七節)")
    print(f"分析期間 2026-07   營業額 {revenue:,.0f} 元   品項 {len(inv)}")

    # ---- Q1 ----
    section("Q1 成本結構 vs 業界標竿")
    st = cost_structure(summary, revenue)
    show = st.copy()
    show["金額"] = show["金額"].map("{:,.0f}".format)
    show["佔營業額"] = show["佔營業額"].map("{:.1%}".format)
    show["超標金額"] = show["超標金額"].map(lambda v: f"{v:,.0f}" if v else "-")
    print(show.to_string(index=False))
    over = st[st["判定"] == "🔴 偏高"]
    if not over.empty:
        print(f"\n  超標項目合計可改善空間 {over['超標金額'].sum():,.0f} 元/月")
    print(f"  淨利率 {summary.get('淨利率', 0):.1%}"
          f"(稅前淨利 {summary.get('稅前淨利(元)', 0):,.0f} 元)")

    # ---- Q2 ----
    section("Q2 食材成本對帳(理論 vs 實際)")
    inv2, rec = food_cost_reconciliation(inv, revenue, actual_purchase)
    print(f"  理論食材成本(真正賣掉的)   {rec['理論食材成本(賣掉的)']:>10,.0f} 元"
          f"   成本率 {rec['理論成本率']:.1%}")
    print(f"  實際叫貨金額(這個月付的)   {rec['實際叫貨金額(付出去的)']:>10,.0f} 元"
          f"   成本率 {rec['叫貨成本率']:.1%}")
    print(f"  差額                        "
          f"{rec['實際叫貨金額(付出去的)'] - rec['理論食材成本(賣掉的)']:>10,.0f} 元")
    print(f"    ├ 庫存變動(錢變成庫存)   {rec['庫存變動']:>10,.0f} 元")
    print(f"    └ 耗損成本                {rec['耗損成本']:>10,.0f} 元")
    print(f"\n  報表上的食材成本率是 {actual_purchase / revenue:.2%}(用叫貨金額算),")
    print(f"  但真正賣掉的東西成本率只有 {rec['理論成本率']:.2%} ——")
    print(f"  差距 {(actual_purchase / revenue) - rec['理論成本率']:.2%} 是「貨還在倉裡」,不是成本失控。")
    lo, hi = FOOD_COST_BENCHMARK
    verdict = "區間內 🟡" if rec["理論成本率"] <= hi else "偏高 🔴"
    print(f"  以理論成本率對標業界 {lo:.0%}~{hi:.0%} -> {verdict}")

    # ---- Q3 ----
    section("Q3 品項毛利與成本效率")
    im = item_margin(inv2)
    cols = ["sku_id", "category", "qty_sold", "unit_cost", "price",
            "margin_rate", "total_margin", "cost_share", "margin_share",
            "efficiency"]
    s = im[cols].copy()
    s.columns = ["品項", "分類", "銷售量", "成本", "售價", "毛利率",
                 "毛利額", "成本佔比", "毛利佔比", "備註"]
    s["毛利率"] = s["毛利率"].map("{:.0%}".format)
    s["毛利額"] = s["毛利額"].map("{:,.0f}".format)
    s["成本佔比"] = s["成本佔比"].map("{:.1%}".format)
    s["毛利佔比"] = s["毛利佔比"].map("{:.1%}".format)
    print(s.to_string(index=False))
    bad = im[im["efficiency"] != ""]
    if not bad.empty:
        print(f"\n  ⚠️ {len(bad)} 個品項的成本佔比高於毛利佔比:"
              f"{', '.join(bad['sku_id'])}")
        print("     這些品項吃掉的食材成本比它們貢獻的毛利多 -> 檢視售價或配方成本")

    # ---- Q4 ----
    section("Q4 供應商評分(規格書 4.4)")
    sc = score_suppliers(supplier)
    print(sc[["rank", "supplier_id", "category", "lead_time_avg",
              "lead_time_std", "payment_term_days", "score"]].to_string(index=False))
    print("\n  ❌ 這張表的分數**不可用於決策**。")
    print("     規格書 4.4 的四個輸入指標,目前只有帳期是真實值:")
    print("       on_time_rate          需要驗收紀錄 -> 無")
    print("       defect_rate           需要退貨紀錄 -> 無")
    print("       price_competitiveness 需要兩家以上比價 -> 無")
    print("     -> 對應工作任務清單 P1-03。本模組先定好公式與輸入契約,")
    print("        採購補資料時才知道要補成什麼格式。")
    print("\n  跨店聯合採購最適化(4.4 的 MILP)未實作:目前只有 1 家分店,")
    print("  最適化問題退化為「照需求叫貨」,沒有跨店調撥的決策空間。")

    # ---- 輸出 ----
    st.to_csv(f"{out_dir}/cost_structure.csv", index=False, encoding="utf-8-sig")
    im.to_csv(f"{out_dir}/item_margin.csv", index=False, encoding="utf-8-sig")
    sc.to_csv(f"{out_dir}/supplier_score.csv", index=False, encoding="utf-8-sig")
    print("\n已輸出 output/cost_structure.csv、item_margin.csv、supplier_score.csv")
    return st, im, sc


if __name__ == "__main__":
    main()
