"""
Two honest models on the Abu Dhabi Police accident data.
  A) Daily accident-count forecast per region (train 2022-2024, test 2025)
  B) Injury severity on record-level injury sheet (1 injured vs 2+), time split
Run: python3 train_models.py   (needs out/accidents_clean.parquet from prepare.py)
"""
import pandas as pd, numpy as np, lightgbm as lgb, json, os
from sklearn.metrics import mean_absolute_error, roc_auc_score, classification_report

df = pd.read_parquet("out/accidents_clean.parquet")
df = df[~df.is_duplicate].copy()
df["n_accidents"] = pd.to_numeric(df.n_accidents, errors="coerce").fillna(1)
df["year"] = pd.to_numeric(df.year, errors="coerce")
df["date"] = pd.to_datetime(df.date)
# drop Feb-2024 storm bulk claims (3 rows, 7,277 "accidents")
storm = (df.date == "2024-02-12") & (df.n_accidents > 500)
print(f"Dropping {storm.sum()} storm bulk-claim rows ({int(df[storm].n_accidents.sum())} claims)")
df = df[~storm]
os.makedirs("out", exist_ok=True); results = {}

# ---------------- A) DAILY COUNT FORECAST ----------------
daily = df.groupby(["region", "date"]).n_accidents.sum().reset_index(name="y")
# complete calendar per region
idx = pd.MultiIndex.from_product([daily.region.unique(), pd.date_range("2022-01-01", "2025-12-31")], names=["region", "date"])
daily = daily.set_index(["region", "date"]).reindex(idx, fill_value=0).reset_index()
daily["dow"] = daily.date.dt.dayofweek; daily["month"] = daily.date.dt.month
daily["doy"] = daily.date.dt.dayofyear; daily["year"] = daily.date.dt.year
daily["weekend"] = daily.dow.isin([5, 6]).astype(int)   # UAE weekend Sat-Sun
# Ramadan windows (approx, Gregorian)
ram = [("2022-04-02","2022-05-01"),("2023-03-23","2023-04-20"),("2024-03-11","2024-04-09"),("2025-03-01","2025-03-29")]
daily["ramadan"] = 0
for a, b in ram: daily.loc[daily.date.between(a, b), "ramadan"] = 1
# UAE public holidays (observed), 2022-2025
hol = [("2022-01-01","2022-01-01"),("2022-05-02","2022-05-05"),("2022-07-08","2022-07-11"),("2022-07-30","2022-07-30"),("2022-10-08","2022-10-08"),("2022-12-01","2022-12-03"),
       ("2023-01-01","2023-01-01"),("2023-04-20","2023-04-23"),("2023-06-27","2023-06-30"),("2023-07-21","2023-07-21"),("2023-09-29","2023-09-29"),("2023-12-01","2023-12-03"),
       ("2024-01-01","2024-01-01"),("2024-04-08","2024-04-12"),("2024-06-15","2024-06-18"),("2024-07-07","2024-07-07"),("2024-09-15","2024-09-15"),("2024-12-01","2024-12-03"),
       ("2025-01-01","2025-01-01"),("2025-03-30","2025-04-01"),("2025-06-05","2025-06-08"),("2025-06-26","2025-06-26"),("2025-09-05","2025-09-05"),("2025-12-01","2025-12-03")]
daily["holiday"] = 0
for a, b in hol: daily.loc[daily.date.between(a, b), "holiday"] = 1
daily["eid"] = 0
for a, b in [h for h in hol if h[0][5:7] in ("04","05","06","07") and h[0] != h[1]]: daily.loc[daily.date.between(a, b), "eid"] = 1
daily["day_after_hol"] = daily.groupby("region").holiday.shift(1).fillna(0)
# weather (Open-Meteo ERA5 archive, daily, one grid point per region)
wx = pd.read_csv("data/weather_daily.csv", parse_dates=["date"])
wx = wx.rename(columns={"temperature_2m_max":"tmax","temperature_2m_min":"tmin","precipitation_sum":"rain","wind_speed_10m_max":"wind","wind_gusts_10m_max":"gust","relative_humidity_2m_mean":"rh","cloud_cover_mean":"cloud","weather_code":"wcode"})
daily = daily.merge(wx[["region","date","tmax","tmin","rain","wind","gust","rh","cloud","wcode"]], on=["region","date"], how="left")
daily["rainy"] = (daily.rain > 1).astype(int); daily["rain_prev"] = daily.groupby("region").rain.shift(1).fillna(0)
daily["fog_risk"] = ((daily.rh > 85) & (daily.wind < 12)).astype(int)   # humid, calm: fog mornings
WX = ["tmax","tmin","rain","rainy","rain_prev","wind","gust","rh","cloud","fog_risk"]
# lag features (within region)
daily = daily.sort_values(["region", "date"])
for L in [1, 7, 14, 28]:
    daily[f"lag{L}"] = daily.groupby("region").y.shift(L)
daily["roll7"] = daily.groupby("region").y.transform(lambda s: s.shift(1).rolling(7).mean())
daily["roll28"] = daily.groupby("region").y.transform(lambda s: s.shift(1).rolling(28).mean())
daily["region"] = daily.region.astype("category")
feats0 = ["region", "dow", "month", "doy", "weekend", "ramadan", "holiday", "eid", "day_after_hol", "lag1", "lag7", "lag14", "lag28", "roll7", "roll28"]
feats = feats0 + WX
tr = daily[(daily.year < 2025) & daily.lag28.notna()]; te = daily[daily.year == 2025]
m0 = lgb.LGBMRegressor(n_estimators=800, learning_rate=0.03, num_leaves=31, objective="poisson", verbose=-1).fit(tr[feats0], tr.y, categorical_feature=["region"])
m = lgb.LGBMRegressor(n_estimators=800, learning_rate=0.03, num_leaves=31, objective="poisson", verbose=-1).fit(tr[feats], tr.y, categorical_feature=["region"])
te = te.assign(pred0=m0.predict(te[feats0]), pred=m.predict(te[feats]))
print("Weather effect on 2025 MAE (no weather -> with weather):")
for r, g in te.groupby("region", observed=True): print(f"  {r:10s} {mean_absolute_error(g.y,g.pred0):.1f} -> {mean_absolute_error(g.y,g.pred):.1f}")
rainy = te[te.rainy==1]; print(f"  rainy days only (n={len(rainy)}): {mean_absolute_error(rainy.y,rainy.pred0):.1f} -> {mean_absolute_error(rainy.y,rainy.pred):.1f}")
print("\n=== A) Daily accident forecast, test year 2025 ===")
rows = []
for r, g in te.groupby("region", observed=True):
    mae = mean_absolute_error(g.y, g.pred); naive = mean_absolute_error(g.y, g.lag7)
    mape = (abs(g.y - g.pred) / g.y.clip(lower=1)).mean() * 100
    rows.append(dict(region=r, mean_daily=round(g.y.mean(), 1), MAE=round(mae, 1), MAPE=round(mape, 1), naive_lag7_MAE=round(naive, 1)))
A = pd.DataFrame(rows); print(A.to_string(index=False)); results["daily_forecast"] = A.to_dict("records")
print("\nFeature importance:", dict(sorted(zip(feats, m.feature_importances_), key=lambda x: -x[1])[:6]))
te[["region", "date", "y", "pred", "pred0", "rain"]].to_csv("out/forecast_2025.csv", index=False)
# residuals by road are computed in build_hotspot_data.py; save daily residuals here
te.assign(resid=te.y-te.pred)[["region","date","y","pred","resid","rain"]].to_csv("out/residuals_2025.csv", index=False)
# weekday / Ramadan effects (descriptive)
print("\nAvg daily accidents by weekday (Mon=0):"); print(daily.groupby("dow").y.mean().round(1).to_string())
print(f"Holiday vs normal daily mean: {daily[daily.holiday==1].y.mean():.1f} vs {daily[daily.holiday==0].y.mean():.1f}")
print(f"Ramadan vs non-Ramadan daily mean: {daily[daily.ramadan==1].y.mean():.1f} vs {daily[daily.ramadan==0].y.mean():.1f}")

# ---------------- B) INJURY SEVERITY ----------------
inj = df[df.severity == "injury"].copy()
inj["n_injured"] = pd.to_numeric(inj.n_injured, errors="coerce")
inj = inj.dropna(subset=["n_injured"])
inj["multi"] = (inj.n_injured >= 2).astype(int)
inj["hour"] = pd.to_numeric(inj.hour, errors="coerce")
inj["night"] = inj.hour.between(22, 23) | inj.hour.between(0, 5)
CAT = ["region", "collision_type", "reason", "surface", "weather", "street"]
for c in CAT: inj[c] = inj[c].astype("category")
NUM = ["hour", "dow", "month", "night"]
trB = inj[inj.year < 2025]; teB = inj[inj.year == 2025]
cb = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=30, is_unbalance=True, verbose=-1)
cb.fit(trB[CAT + NUM], trB.multi, categorical_feature=CAT)
pr = cb.predict_proba(teB[CAT + NUM])[:, 1]
auc = roc_auc_score(teB.multi, pr)
print(f"\n=== B) Injury severity: 2+ injured vs 1 (train 2022-24 n={len(trB)}, test 2025 n={len(teB)}) ===")
print(f"Base rate 2+ injured: {teB.multi.mean():.2%}   AUC: {auc:.3f}")
print(classification_report(teB.multi, (pr > 0.5).astype(int), digits=3))
imp = pd.Series(cb.feature_importances_, index=CAT + NUM).sort_values(ascending=False)
print("Feature importance:\n" + imp.to_string())
results["injury_severity"] = dict(auc=round(auc, 3), n_train=len(trB), n_test=len(teB), base_rate=round(teB.multi.mean(), 3))
# which collision types carry most multi-injury risk
print("\nShare with 2+ injured by collision type (all years, n>=50):")
ct = inj.groupby("collision_type", observed=True).agg(n=("multi", "size"), multi_rate=("multi", "mean"))
print(ct[ct.n >= 50].sort_values("multi_rate", ascending=False).round(3).to_string())
print("\nInjury accidents by hour:"); print(inj.hour.value_counts().sort_index().to_string())
json.dump(results, open("out/results.json", "w"), indent=2)
