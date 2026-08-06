"""
模組三:高酒水潛力桌預測(監督式學習)

商業問題
--------
「客人剛入座,外場要不要主動送酒單、開口介紹?」

這是餐酒館外場每天做幾十次的判斷。做對了,客單價明顯上升;
做錯了 —— 對明顯不喝酒的桌硬推,客人會覺得被推銷,體驗變差。

現在這個判斷靠的是資深外場的直覺。這個模型的目的是把那個直覺量化,
讓新人也能有依據。

重要的設計決定:只用「入座當下就知道的資訊」
--------------------------------------------
特徵只包含:人數、到店時間、週幾、是否常客。

刻意不使用「停留時間」與「已點餐點」,因為那些是入座之後才會知道的。
如果拿它們來訓練,模型分數會很漂亮,但實際上根本不能用 ——
外場需要的是「客人剛坐下」那一刻的判斷。

這種「資料外洩(data leakage)」是機器學習專題最常見的錯誤,
面試時能主動說明自己避開了這個坑,比模型準確率高幾個百分點更有價值。

預測目標
--------
酒水佔比 >= 40% 視為「高酒水潛力桌」。
40% 是餐酒館的營運目標線 —— 低於這個數字,這門生意的獲利模式就不成立。
"""

import os
import sys
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE_DIR, "data")
OUT = os.path.join(BASE_DIR, "output")

TARGET_RATIO = 0.40   # 酒水佔比目標線

# 只用入座當下已知的資訊(避免資料外洩)
FEATURES = ["party_size", "arrival_hour", "is_buzz_hour", "is_weekend", "is_repeat"]
FEATURE_LABELS = {
    "party_size": "同桌人數",
    "arrival_hour": "到店時間",
    "is_buzz_hour": "微醺時段入座(21點後)",
    "is_weekend": "週五六",
    "is_repeat": "常客回訪",
}


def load():
    v = pd.read_csv(os.path.join(DATA, "visits.csv"))
    v["is_high_drink"] = (v["drink_ratio"] >= TARGET_RATIO).astype(int)
    return v


def main():
    os.makedirs(OUT, exist_ok=True)
    v = load()

    X = v[FEATURES]
    y = v["is_high_drink"]

    print("=" * 74)
    print("高酒水潛力桌預測")
    print("=" * 74)
    print(f"樣本數:{len(v):,} 桌")
    print(f"高酒水桌(酒水佔比 >= {TARGET_RATIO:.0%}):{y.mean():.1%}")
    print(f"使用特徵(皆為入座當下已知):{', '.join(FEATURE_LABELS[f] for f in FEATURES)}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # ---------------- 邏輯回歸 ----------------
    scaler = StandardScaler().fit(X_tr)
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(scaler.transform(X_tr), y_tr)

    lr_pred = lr.predict(scaler.transform(X_te))
    lr_prob = lr.predict_proba(scaler.transform(X_te))[:, 1]
    lr_auc = roc_auc_score(y_te, lr_prob)

    print("\n" + "-" * 74)
    print("模型 A:邏輯回歸")
    print("-" * 74)
    print(classification_report(y_te, lr_pred, target_names=["一般桌", "高酒水桌"], digits=3))
    print(f"ROC-AUC:{lr_auc:.3f}")

    cm = confusion_matrix(y_te, lr_pred)
    print("\n混淆矩陣")
    print(pd.DataFrame(
        cm,
        index=["實際:一般桌", "實際:高酒水"],
        columns=["預測:一般桌", "預測:高酒水"],
    ).to_string())

    # ---- 係數轉成勝算比,這是能拿去跟店長講的語言 ----
    print("\n各因素對「高酒水消費」的影響(勝算比 Odds Ratio)")
    print("勝算比 > 1 表示提高機率,< 1 表示降低")
    odds = pd.DataFrame({
        "因素": [FEATURE_LABELS[f] for f in FEATURES],
        "係數": lr.coef_[0].round(3),
        "勝算比": np.exp(lr.coef_[0]).round(3),
    }).sort_values("勝算比", ascending=False)
    print(odds.to_string(index=False))

    # ---------------- 決策樹 ----------------
    dt = DecisionTreeClassifier(
        max_depth=4, min_samples_leaf=40, random_state=42
    ).fit(X_tr, y_tr)

    dt_prob = dt.predict_proba(X_te)[:, 1]
    dt_auc = roc_auc_score(y_te, dt_prob)

    print("\n" + "-" * 74)
    print("模型 B:決策樹")
    print("-" * 74)
    print(classification_report(y_te, dt.predict(X_te),
                               target_names=["一般桌", "高酒水桌"], digits=3))
    print(f"ROC-AUC:{dt_auc:.3f}")

    print("\n特徵重要性")
    imp = pd.DataFrame({
        "因素": [FEATURE_LABELS[f] for f in FEATURES],
        "重要性": dt.feature_importances_.round(3),
    }).sort_values("重要性", ascending=False)
    print(imp.to_string(index=False))

    print("\n決策規則(可以直接印出來貼在外場)")
    print(export_text(dt, feature_names=[FEATURE_LABELS[f] for f in FEATURES],
                      max_depth=3, show_weights=False))

    # ---------------- 模型選擇 ----------------
    print("-" * 74)
    print(f"模型比較:邏輯回歸 AUC {lr_auc:.3f}  vs  決策樹 AUC {dt_auc:.3f}")
    better = "邏輯回歸" if lr_auc >= dt_auc else "決策樹"
    print(f"選用:{better}")
    print("說明:AUC 相近時優先選邏輯回歸,因為它的勝算比可以直接解釋給店長聽,")
    print("      而營運場景中『能被理解與信任』比多零點幾的準確率重要。")

    # ---------------- 轉成營運行動 ----------------
    print("\n" + "=" * 74)
    print("營運應用:哪些桌該優先主動推酒")
    print("=" * 74)

    v_all = v.copy()
    v_all["prob"] = lr.predict_proba(scaler.transform(X))[:, 1]

    grid = (v_all.groupby(["is_buzz_hour", "is_repeat"])
                 .agg(桌數=("visit_id", "count"),
                      平均高酒水機率=("prob", "mean"),
                      實際酒水佔比=("drink_ratio", "mean"),
                      平均客單價=("spend_per_head", "mean"))
                 .reset_index())
    grid["時段"] = grid["is_buzz_hour"].map({0: "早場 17-20", 1: "微醺 21+"})
    grid["客群"] = grid["is_repeat"].map({0: "新客", 1: "常客"})
    grid["平均高酒水機率"] = (grid["平均高酒水機率"] * 100).round(1).astype(str) + "%"
    grid["實際酒水佔比"] = (grid["實際酒水佔比"] * 100).round(1).astype(str) + "%"
    grid["平均客單價"] = grid["平均客單價"].round(0).astype(int)

    print(grid[["時段", "客群", "桌數", "平均高酒水機率",
                "實際酒水佔比", "平均客單價"]].to_string(index=False))

    top = grid.sort_values("平均客單價", ascending=False).iloc[0]
    print(f"\n結論:『{top['時段']} + {top['客群']}』是客單價最高的組合"
          f"(平均 {top['平均客單價']} 元/人)")
    print("行動建議:這個組合的桌,入座時直接送上酒單並主動介紹當日推薦酒款;")
    print("          相對地,早場新客先以餐點服務為主,避免一入座就推銷造成反感。")

    # ---------------- 輸出 ROC 圖 ----------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fpr_l, tpr_l, _ = roc_curve(y_te, lr_prob)
        fpr_d, tpr_d, _ = roc_curve(y_te, dt_prob)

        plt.figure(figsize=(6, 5))
        plt.plot(fpr_l, tpr_l, label=f"Logistic Regression (AUC={lr_auc:.3f})")
        plt.plot(fpr_d, tpr_d, label=f"Decision Tree (AUC={dt_auc:.3f})")
        plt.plot([0, 1], [0, 1], "k--", lw=0.8, label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC - High Drink-Ratio Table Prediction")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, "roc_upsell.png"), dpi=120)
        plt.close()
        print("\nROC 曲線已輸出:output/roc_upsell.png")
    except Exception as e:
        print(f"\n(繪圖略過:{e})")


if __name__ == "__main__":
    main()
