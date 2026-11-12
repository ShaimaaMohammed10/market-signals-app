"""Run once: builds everything the Streamlit app reads from outputs/ (metrics, predictions, final model)."""
import warnings, joblib, json
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import forecast_utils as fu

warnings.filterwarnings("ignore")
OUT = fu.BASE / "outputs"; OUT.mkdir(exist_ok=True)
RS = 42

data = fu.load_clean()
feats = fu.build_features(data)
cols = fu.feature_columns(feats)
train_all = fu.training_rows(feats)
num = [c for c in cols if c != "category"]

def preproc():
    return ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), ["category"])])

dates = np.array(sorted(train_all.year_month.unique()))
split = dates[int(len(dates) * 0.8)]
tr, te = train_all[train_all.year_month < split], train_all[train_all.year_month >= split]
print("split", pd.Timestamp(split).date(), len(tr), len(te))

def metrics(name, y, p):
    return {"Model": name, "MAE": mean_absolute_error(y, p), "RMSE": mean_squared_error(y, p) ** .5, "R2": r2_score(y, p)}

TUNED_RF = dict(n_estimators=400, min_samples_leaf=1, max_features=0.3, max_depth=None, random_state=RS, n_jobs=-1)
def models():
    return {"Linear Regression": LinearRegression(), "Ridge": Ridge(alpha=10.0),
            "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=RS, n_jobs=-1),
            "Random Forest (tuned)": RandomForestRegressor(**TUNED_RF),
            "Hist Gradient Boosting": HistGradientBoostingRegressor(max_iter=150, random_state=RS)}

def run(target, cur, extra=None):
    rows = [metrics("Baseline: last month", te[target], te[cur])]
    if extra: rows += [metrics(n, te[target], te[c]) for n, c in extra.items()]
    fitted = {}
    for n, m in models().items():
        p = Pipeline([("preprocessing", preproc()), ("model", m)]).fit(tr[cols], tr[target])
        fitted[n] = p; rows.append(metrics(n, te[target], p.predict(te[cols])))
    return pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True), fitted

sales_tbl, sales_fit = run("target_next_sales", "dc_total_sales", {"Baseline: avg last 3 months": "sales_roll3"})
late_tbl, _ = run("target_next_late_rate", "dc_late_rate")
sales_tbl.round(4).to_csv(OUT / "model_comparison_sales.csv", index=False)
late_tbl.round(4).to_csv(OUT / "model_comparison_late_rate.csv", index=False)

# test-period predictions (train-only model) for backtests/plots
best = sales_fit["Random Forest (tuned)"]
tp = te[["category", "year_month", "target_next_sales"]].copy()
tp["target_month"] = tp.year_month + pd.offsets.MonthBegin(1)
tp["predicted"] = np.clip(best.predict(te[cols]), 0, None)
tp["baseline_last_month"] = te["dc_total_sales"].values
tp.rename(columns={"target_next_sales": "actual"}).drop(columns="year_month").to_csv(OUT / "test_predictions.csv", index=False)

# error by category
ce = tp.groupby("category").apply(lambda g: pd.Series({
    "test_months": len(g), "avg_actual_sales": g.target_next_sales.mean(),
    "MAE": mean_absolute_error(g.target_next_sales, g.predicted),
    "MAPE_pct": float(np.mean(np.abs(g.target_next_sales - g.predicted) / g.target_next_sales) * 100)})).reset_index()
ce.round(2).to_csv(OUT / "category_errors.csv", index=False)

# permutation importance
pi = permutation_importance(best, te[cols], te["target_next_sales"], n_repeats=15, random_state=RS,
                            scoring="neg_mean_absolute_error")
pd.Series(pi.importances_mean, index=cols, name="mae_increase").sort_values(ascending=False).round(2)\
  .to_csv(OUT / "feature_importance.csv", header=True, index_label="feature")

# FINAL model: same tuned RF but trained on ALL available rows (used only for the app forecast)
final = Pipeline([("preprocessing", preproc()), ("model", RandomForestRegressor(**TUNED_RF))])
final.fit(train_all[cols], train_all["target_next_sales"])
joblib.dump({"model": final, "feature_columns": cols, "name": "Random Forest (tuned), trained on all rows",
             "trained_until": str(train_all.year_month.max().date()),
             "forecast_month": str(fu.FORECAST_MONTH.date())}, OUT / "app_model.joblib")

# =========================================================
# Prediction intervals: rolling-origin split-conformal on log(actual / predicted), by size group.
# Calibration residuals are out-of-sample (the model is refit before each of the last 6 train months).
# =========================================================
SIZE_CUT = 10_000
size_tr = tr.groupby("category").dc_total_sales.mean()
size_all = data.groupby("category").dc_total_sales.mean()
grp_tr = lambda c: "large" if size_tr[c] >= SIZE_CUT else "small"
assert all((size_tr[c] >= SIZE_CUT) == (size_all[c] >= SIZE_CUT) for c in size_all.index), "size groups differ between periods"
grp_all = lambda c: "large" if size_all[c] >= SIZE_CUT else "small"

res = []
for k in range(6, 0, -1):
    m0 = pd.Timestamp(split) - pd.DateOffset(months=k)
    fit, c = tr[tr.year_month < m0], tr[tr.year_month == m0]
    m = Pipeline([("preprocessing", preproc()),
                  ("model", RandomForestRegressor(**{**TUNED_RF, "n_estimators": 200}))]).fit(fit[cols], fit.target_next_sales)
    p = np.clip(m.predict(c[cols]), 1, None)
    res.append(c.assign(r=np.log(c.target_next_sales / p)))
cal = pd.concat(res); cal["g"] = cal.category.map(grp_tr)
ALPHA = 0.10
q = {g: [float(cal[cal.g == g].r.quantile(ALPHA / 2)), float(cal[cal.g == g].r.quantile(1 - ALPHA / 2))] for g in ("large", "small")}

# honest coverage check on the untouched test period (train-only model)
pt = np.clip(best.predict(te[cols]), 1, None)
tg = te.category.map(grp_tr)
lo = pt * np.exp(tg.map(lambda g: q[g][0]).values); hi = pt * np.exp(tg.map(lambda g: q[g][1]).values)
hit = ((te.target_next_sales.values >= lo) & (te.target_next_sales.values <= hi))
cover = pd.DataFrame([{"group": g, "n_calibration": int((cal.g == g).sum()), "n_test": int((tg == g).sum()),
                       "nominal": 1 - ALPHA, "actual_coverage": float(hit[(tg == g).values].mean()),
                       "mean_relative_width": float(((hi - lo) / pt)[(tg == g).values].mean())} for g in ("large", "small")])
cover.round(3).to_csv(OUT / "interval_coverage.csv", index=False)
print(cover.round(3).to_string(index=False))

# =========================================================
# Final forecast table (final model on all rows) with intervals for the "large" group only
# =========================================================
final_rows = fu.forecast_rows(feats, cols)
final_rows["forecast"] = np.clip(final.predict(final_rows[cols]), 0, None)
ft = final_rows[["category", "dc_total_sales", "forecast"]].rename(columns={"dc_total_sales": "last_actual"})
ft["size_group"] = ft.category.map(grp_all)
ft["lower"] = [f * np.exp(q["large"][0]) if g == "large" else np.nan for f, g in zip(ft.forecast, ft.size_group)]
ft["upper"] = [f * np.exp(q["large"][1]) if g == "large" else np.nan for f, g in zip(ft.forecast, ft.size_group)]
ft.round(1).to_csv(OUT / "forecast_table.csv", index=False)

# =========================================================
# Category segments (KMeans, k chosen by silhouette) on sales level + volatility
# =========================================================
gg = data.groupby("category")
seg = pd.DataFrame({"avg_monthly_sales": gg.dc_total_sales.mean(),
                    "cv_sales": gg.dc_total_sales.std() / gg.dc_total_sales.mean(),
                    "avg_profit_ratio": gg.dc_avg_profit_ratio.mean(), "avg_late_rate": gg.dc_late_rate.mean(),
                    "avg_discount": gg.dc_avg_discount.mean(), "months": gg.size()})
Z = StandardScaler().fit_transform(np.column_stack([np.log(seg.avg_monthly_sales), seg.cv_sales]))
sil = {k: float(silhouette_score(Z, KMeans(k, n_init=20, random_state=RS).fit_predict(Z))) for k in (2, 3, 4)}
K = 3
lab = KMeans(K, n_init=20, random_state=RS).fit_predict(Z)
order = pd.Series(seg.avg_monthly_sales.values).groupby(lab).mean().sort_values(ascending=False).index.tolist()
names = {order[0]: "Large & stable", order[1]: "Mid-size", order[2]: "Small & volatile"}
seg["segment"] = [names[l] for l in lab]
seg["size_group"] = seg.index.map(grp_all)
seg.reset_index().round(4).sort_values("avg_monthly_sales", ascending=False).to_csv(OUT / "category_segments.csv", index=False)
print(seg.groupby("segment").agg(n=("months", "size"), avg_sales=("avg_monthly_sales", "mean"), cv=("cv_sales", "mean"))
      .round(2).to_string()); print("silhouette:", {k: round(v, 3) for k, v in sil.items()})

json.dump({"model_name": "Random Forest (tuned), trained on all rows", "trained_until": str(train_all.year_month.max().date()),
           "forecast_month": str(fu.FORECAST_MONTH.date()), "interval_level": 1 - ALPHA, "size_cut": SIZE_CUT,
           "log_ratio_quantiles": q, "silhouette": sil, "k": K}, open(OUT / "interval_info.json", "w"), indent=2)

print("assets saved"); print(sales_tbl.round(1).to_string(index=False))
