"""
Baseline models on the cleaned accident table.

    python train_baseline.py

Model A: severity classification (injury / no_injury / hit_and_run)
Model B: number-of-injured regression on injury rows only
Requires: pip install pandas pyarrow scikit-learn lightgbm
"""
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_absolute_error
import lightgbm as lgb

df = pd.read_parquet("out/accidents_clean.parquet")
df = df[~df.is_duplicate]

CAT = [c for c in ["region", "city", "collision_type", "reason", "surface", "weather", "kind"] if c in df]
NUM = [c for c in ["hour", "dow", "month", "year"] if c in df]
for c in CAT:
    df[c] = df[c].astype("category")

# ---- Model A: severity ----
X = df[CAT + NUM]; y = df["severity"].astype("category")
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
clf = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, class_weight="balanced", verbose=-1)
clf.fit(Xtr, ytr, categorical_feature=CAT)
print("=== Severity classifier ===")
print(classification_report(yte, clf.predict(Xte)))
imp = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
print(imp.to_string(), "\n")

# ---- Model B: injured count (injury sheet only) ----
inj = df[(df.severity == "injury") & df["n_injured"].notna()].copy()
if len(inj) > 500:
    inj["n_injured"] = pd.to_numeric(inj["n_injured"], errors="coerce")
    inj = inj.dropna(subset=["n_injured"])
    X = inj[CAT + NUM]; y = inj["n_injured"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    reg = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, objective="poisson", verbose=-1)
    reg.fit(Xtr, ytr, categorical_feature=CAT)
    pred = reg.predict(Xte)
    print("=== Injured-count regression ===")
    print(f"MAE {mean_absolute_error(yte, pred):.3f}  | baseline (predict mean) MAE {mean_absolute_error(yte, np.full_like(yte, ytr.mean())):.3f}")
