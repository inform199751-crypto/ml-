"""
模組四:客群分群(非監督式學習)

商業問題
--------
「我們的常客其實不是同一種人 —— 該怎麼分,才能對不同的人做不同的事?」

餐酒館常見的錯誤是把所有客人當成一種人:同一份 DM、同一個優惠、同一套話術。
實際上「來喝酒的」和「來吃飯的」需要的東西完全不同:
對酒客推套餐折扣沒用,對餐客推調酒買一送一也沒用。

這個模組用 K-means 把客人自動分群,再用餐飲的語言解讀每一群是什麼人,
最後給出各群對應的行動建議。

技術重點
--------
  1. K-means 分群 + 用手肘法(Elbow)與輪廓係數(Silhouette)決定 K
  2. 特徵標準化:客單價的數量級是千,酒水佔比是 0~1,
     不標準化的話距離計算會被客單價完全主導,分群結果等於只看客單價
  3. PCA 降維到 2 維畫圖:分群結果本身是 5 維的,人看不到,
     用 PCA 壓到 2 維才能視覺化確認分群是否真的分開了
"""

import os
import sys

# KMeans 在 Windows + MKL 環境會噴記憶體警告,先限制執行緒數(需在 import sklearn 前設定)
os.environ.setdefault("OMP_NUM_THREADS", "3")

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE_DIR, "data")
OUT = os.path.join(BASE_DIR, "output")

MIN_VISITS = 2       # 只分析回訪過的客人(單次客無法看出消費模式)

FEATURES = {
    "visits": "造訪次數",
    "avg_spend": "平均客單價",
    "avg_drink_ratio": "平均酒水佔比",
    "avg_duration": "平均停留時間",
    "avg_hour": "平均到店時間",
}


def build_customer_table(visits):
    """把一桌一列的資料,彙總成一位客人一列。"""
    cust = visits.groupby("customer_id").agg(
        visits=("visit_id", "count"),
        avg_spend=("spend_per_head", "mean"),
        avg_drink_ratio=("drink_ratio", "mean"),
        avg_duration=("duration_min", "mean"),
        avg_hour=("arrival_hour", "mean"),
        total_revenue=("total_revenue", "sum"),
        total_profit=("total_profit", "sum"),
    ).reset_index()
    return cust[cust["visits"] >= MIN_VISITS].copy()


def choose_k(X, k_range=range(2, 8)):
    """用手肘法與輪廓係數挑 K。"""
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
        rows.append({
            "K": k,
            "inertia": round(km.inertia_, 1),
            "silhouette": round(silhouette_score(X, km.labels_), 4),
        })
    return pd.DataFrame(rows)


def name_clusters(profile):
    """
    用餐飲語言給每一群命名。

    這一步是模型做不到的 —— K-means 只會輸出 0/1/2/3,
    「這群是酒客還是餐客」需要人來判讀,而判讀依據來自餐飲現場經驗。

    命名採「群與群相對比較」而非固定門檻:
    分群的本質是把客人分出差異,所以要問的是「這群相對於其他群,特別在哪裡」。
    先前用絕對中位數判斷會出現兩群同名的問題 —— 因為同名就等於沒分群。

    判讀順序(依餐酒館的經營優先度):
      1. 造訪次數最高的  -> 核心常客,店的基本盤
      2. 剩下酒水佔比最高 -> 夜間酒客,毛利貢獻主力
      3. 酒水佔比最低的  -> 餐點取向客,要用不同方式經營
    """
    labels = {}
    remaining = list(profile["cluster"])

    # 1. 最常回訪的一群
    c_vip = profile.loc[profile["造訪次數"].idxmax(), "cluster"]
    labels[c_vip] = (
        "核心常客",
        "回訪頻率明顯高於其他群,是店的基本盤。"
        "維繫成本最低、CP 值最高 —— 記住他們的名字與慣點酒款,比任何折扣都有效",
    )
    remaining.remove(c_vip)

    # 2. 剩下的群裡,酒水佔比最高的
    if remaining:
        sub = profile[profile["cluster"].isin(remaining)]
        c_drink = sub.loc[sub["平均酒水佔比"].idxmax(), "cluster"]
        labels[c_drink] = (
            "夜間酒客",
            "入座時間晚、酒水佔比高、停留久。餐酒館真正的毛利來源,"
            "適合推新調酒與單杯精品酒;推餐點套餐對這群無效",
        )
        remaining.remove(c_drink)

    # 3. 酒水佔比最低的 -> 來吃飯的
    if remaining:
        sub = profile[profile["cluster"].isin(remaining)]
        c_food = sub.loc[sub["平均酒水佔比"].idxmin(), "cluster"]
        labels[c_food] = (
            "餐點取向客",
            "來吃飯而非喝酒,酒水佔比低、入座偏早、停留較短。"
            "對這群該推的是『餐酒搭配套餐』(把酒包進套餐降低決策門檻),而不是單點酒單",
        )
        remaining.remove(c_food)

    for c in remaining:
        labels[c] = ("一般客", "消費模式無明顯特徵,可作為對照基準")

    return labels


def main():
    os.makedirs(OUT, exist_ok=True)
    visits = pd.read_csv(os.path.join(DATA, "visits.csv"))
    cust = build_customer_table(visits)

    print("=" * 78)
    print("客群分群")
    print("=" * 78)
    print(f"總客人數:{visits['customer_id'].nunique():,}")
    print(f"回訪客(造訪 >= {MIN_VISITS} 次):{len(cust):,} 人  <- 分析對象")
    print(f"回訪客貢獻營收佔比:"
          f"{cust['total_revenue'].sum() / visits['total_revenue'].sum():.1%}")

    feat_cols = list(FEATURES.keys())
    X_raw = cust[feat_cols].values

    # 標準化:不做這步,分群會被客單價的數量級主導
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    # ---------------- 決定 K ----------------
    print("\n" + "-" * 78)
    print("決定分幾群(手肘法 + 輪廓係數)")
    print("-" * 78)
    scores = choose_k(X)
    print(scores.to_string(index=False))

    best_k = int(scores.loc[scores["silhouette"].idxmax(), "K"])
    print(f"\n輪廓係數最高的 K = {best_k}")
    # 分群結果要能對應到實際可執行的行動,群數太多營運端無法落實
    k = min(best_k, 4) if best_k > 4 else best_k
    if k != best_k:
        print(f"但實務上採用 K = {k}:分群的目的是要能對應行動,")
        print("分太多群外場與行銷端記不住也做不到,反而失去意義。")

    # ---------------- 分群 ----------------
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
    cust["cluster"] = km.labels_

    profile = cust.groupby("cluster").agg(
        人數=("customer_id", "count"),
        造訪次數=("visits", "mean"),
        平均客單價=("avg_spend", "mean"),
        平均酒水佔比=("avg_drink_ratio", "mean"),
        平均停留=("avg_duration", "mean"),
        平均到店時間=("avg_hour", "mean"),
        總毛利=("total_profit", "sum"),
    ).round(2).reset_index()

    labels = name_clusters(profile)
    profile["客群名稱"] = profile["cluster"].map(lambda c: labels[c][0])
    profile["說明"] = profile["cluster"].map(lambda c: labels[c][1])

    print("\n" + "=" * 78)
    print("分群結果")
    print("=" * 78)
    show = profile.copy()
    show["平均酒水佔比"] = (show["平均酒水佔比"] * 100).round(1).astype(str) + "%"
    show["平均客單價"] = show["平均客單價"].round(0).astype(int)
    show["平均停留"] = show["平均停留"].round(0).astype(int).astype(str) + " 分"
    show["平均到店時間"] = show["平均到店時間"].round(1)
    print(show[["cluster", "客群名稱", "人數", "造訪次數", "平均客單價",
                "平均酒水佔比", "平均停留", "平均到店時間"]].to_string(index=False))

    # ---------------- 毛利貢獻:分群的商業意義 ----------------
    profile["毛利佔比"] = profile["總毛利"] / profile["總毛利"].sum()
    profile["人數佔比"] = profile["人數"] / profile["人數"].sum()
    profile["價值倍數"] = (profile["毛利佔比"] / profile["人數佔比"]).round(2)

    print("\n" + "=" * 78)
    print("各客群的價值(價值倍數 > 1 表示以較少人數貢獻較多毛利)")
    print("=" * 78)
    val = profile[["客群名稱", "人數", "毛利佔比", "人數佔比", "價值倍數"]].copy()
    val["毛利佔比"] = (val["毛利佔比"] * 100).round(1).astype(str) + "%"
    val["人數佔比"] = (val["人數佔比"] * 100).round(1).astype(str) + "%"
    print(val.sort_values("價值倍數", ascending=False).to_string(index=False))

    print("\n" + "=" * 78)
    print("各客群的行動建議")
    print("=" * 78)
    for _, r in profile.sort_values("價值倍數", ascending=False).iterrows():
        print(f"\n【{r['客群名稱']}】{r['人數']} 人,佔毛利 {r['毛利佔比']:.1%}"
              f"(價值倍數 {r['價值倍數']})")
        print(f"  特徵:{r['說明']}")

    # ---------------- PCA 視覺化 ----------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        pca = PCA(n_components=2, random_state=42)
        X2 = pca.fit_transform(X)
        var = pca.explained_variance_ratio_

        plt.figure(figsize=(8, 6))
        for c in sorted(cust["cluster"].unique()):
            m = cust["cluster"] == c
            label = profile.loc[profile["cluster"] == c, "客群名稱"].iloc[0]
            plt.scatter(X2[m, 0], X2[m, 1], alpha=0.6, s=28, label=f"C{c}")
        plt.xlabel(f"PC1 ({var[0]:.1%} variance)")
        plt.ylabel(f"PC2 ({var[1]:.1%} variance)")
        plt.title("Customer Segments (PCA projection)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, "customer_segments.png"), dpi=120)
        plt.close()
        print(f"\n分群圖已輸出:output/customer_segments.png")
        print(f"(PCA 前兩維共解釋 {var.sum():.1%} 的變異)")
    except Exception as e:
        print(f"\n(繪圖略過:{e})")

    cust.to_csv(os.path.join(OUT, "customer_segments.csv"),
                index=False, encoding="utf-8-sig")
    profile.to_csv(os.path.join(OUT, "segment_profile.csv"),
                   index=False, encoding="utf-8-sig")
    print("客群明細已輸出:output/customer_segments.csv")


if __name__ == "__main__":
    main()
