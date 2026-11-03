"""Severity model (2+ injured vs 1) with SDI road attributes joined on street name."""
import json, pandas as pd, numpy as np, lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold

j = json.load(open("data/sdi_roads_match.json", encoding="utf-8"))
match = pd.DataFrame(j["match"]).dropna(subset=["road"])[["street", "road"]]
stats = pd.DataFrame(j["stats"].values()).rename(columns={"NAME_AR": "road"})
stats["rd_class_name"] = stats.rd_class.map({int(k): v for k, v in j["rd_class"].items()})
roads = match.merge(stats, on="road", how="left")

df = pd.read_parquet("out/accidents_clean.parquet")
inj = df[(df.severity == "injury") & ~df.is_duplicate].copy()
inj["n_injured"] = pd.to_numeric(inj.n_injured, errors="coerce"); inj = inj.dropna(subset=["n_injured"])
inj["multi"] = (inj.n_injured >= 2).astype(int)
inj["hour"] = pd.to_numeric(inj.hour, errors="coerce"); inj["year"] = pd.to_numeric(inj.year, errors="coerce")
inj = inj.merge(roads, on="street", how="left")
inj["matched"] = inj.road.notna().astype(int)
inj["rd_class"] = inj.rd_class.fillna(0).astype(int)     # 0 = unmatched
print(f"Injury accidents: {len(inj)} | matched to SDI road: {inj.matched.sum()} ({inj.matched.mean():.0%})")

BASE_CAT = ["region", "collision_type", "reason", "surface", "weather"]
BASE_NUM = ["hour", "dow", "month"]
ROAD = ["rd_class", "lanes", "speed", "avgspeed", "len_m", "n_seg", "matched"]
for c in BASE_CAT + ["street"]: inj[c] = inj[c].astype("category")

def evaluate(feats, cats, label):
    tr, te = inj[inj.year < 2025], inj[inj.year == 2025]
    m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=30, is_unbalance=True, verbose=-1)
    m.fit(tr[feats], tr.multi, categorical_feature=cats)
    p = m.predict_proba(te[feats])[:, 1]
    auc, ap = roc_auc_score(te.multi, p), average_precision_score(te.multi, p)
    # 5-fold CV on all years for a steadier estimate
    cv = []
    for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(inj, inj.multi):
        mm = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=30, is_unbalance=True, verbose=-1)
        mm.fit(inj.iloc[a][feats], inj.multi.iloc[a], categorical_feature=cats)
        cv.append(roc_auc_score(inj.multi.iloc[b], mm.predict_proba(inj.iloc[b][feats])[:, 1]))
    print(f"{label:38s} AUC-2025 {auc:.3f} | AP {ap:.3f} | CV-AUC {np.mean(cv):.3f}±{np.std(cv):.3f}")
    return m

print("\n=== Severity: 2+ injured vs 1 ===")
evaluate(BASE_CAT + BASE_NUM, BASE_CAT, "baseline (no street, no road)")
evaluate(BASE_CAT + BASE_NUM + ["street"], BASE_CAT + ["street"], "+ street name (high-cardinality)")
m = evaluate(BASE_CAT + BASE_NUM + ROAD, BASE_CAT, "+ SDI road attributes")
imp = pd.Series(m.feature_importances_, index=BASE_CAT + BASE_NUM + ROAD).sort_values(ascending=False)
print("\nFeature importance (road model):\n" + imp.to_string())

# matched-only subset: does road class matter?
sub = inj[inj.matched == 1]
print("\nMulti-injury rate by road class (matched only):")
print(sub.groupby("rd_class_name").agg(n=("multi", "size"), multi_rate=("multi", "mean"), avg_speed=("speed", "mean")).query("n>=40").sort_values("multi_rate", ascending=False).round(3).to_string())

# ---- street risk table: injuries per km per year ----
yrs = inj.year.nunique()
risk = (inj[inj.matched == 1].groupby("road").agg(accidents=("multi", "size"), injured=("n_injured", "sum"), multi_rate=("multi", "mean"))
        .merge(stats.set_index("road")[["rd_class_name", "lanes", "speed", "len_m"]], left_index=True, right_index=True))
risk["km"] = risk.len_m / 1000 / 2          # both carriageways counted in SDI → approx single-direction length
risk["inj_acc_per_km_yr"] = risk.accidents / risk.km / yrs
risk = risk.sort_values("inj_acc_per_km_yr", ascending=False)
risk.round(2).to_csv("out/road_risk.csv", encoding="utf-8")
print("\nTop roads by injury accidents per km per year (min 15 accidents):")
print(risk[risk.accidents >= 15].head(15)[["accidents", "injured", "multi_rate", "rd_class_name", "speed", "km", "inj_acc_per_km_yr"]].round(2).to_string())
