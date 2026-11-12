import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import forecast_utils as fu

st.set_page_config(page_title="Market Signals — Sales Forecast", page_icon="📈", layout="wide")
OUT = fu.BASE / "outputs"
SMALL_SALES, HIGH_MAPE = 10_000, 25
SEG_COLORS = {"Large & stable": "#1f4e79", "Mid-size": "#d98c00", "Small & volatile": "#c0392b"}


# ---------- cached loaders (the app only reads precomputed files: no model needed at runtime) ----------
@st.cache_data
def load_data():
    return fu.load_clean()

@st.cache_data
def load_csv(name, **kw):
    return pd.read_csv(OUT / name, **kw)

@st.cache_data
def load_info():
    return json.load(open(OUT / "interval_info.json"))


data = load_data()
info = load_info()
ft = load_csv("forecast_table.csv").set_index("category")
seg = load_csv("category_segments.csv").set_index("category")
errors = load_csv("category_errors.csv").set_index("category")
coverage = load_csv("interval_coverage.csv")
test_pred = load_csv("test_predictions.csv", parse_dates=["target_month"])
LEVEL = f"{info['interval_level']:.0%}"

st.sidebar.title("📈 Market Signals")
page = st.sidebar.radio("Page", ["Forecast", "Category segments", "Category overview", "Model performance", "Data notes (عربي)"])
st.sidebar.caption(f"Data: {data.year_month.min():%Y-%m} → {fu.LAST_COMPLETE_MONTH:%Y-%m}  \n"
                   f"Forecast month: **{fu.FORECAST_MONTH:%Y-%m}**  \n"
                   f"Model: {info['model_name']}")


def fmt(x):
    return f"{x:,.0f}"


# =========================================================
if page == "Forecast":
    st.title("Next-month sales forecast")
    st.caption(f"Predicts total sales for **{fu.FORECAST_MONTH:%B %Y}** — the month right after the last complete month in the data. "
               "It is not a live forecast of today.")

    cats = sorted(data.category.unique())
    cat = st.selectbox("Category", cats, index=cats.index("Fishing") if "Fishing" in cats else 0)
    hist = data[data.category == cat].set_index("year_month").dc_total_sales
    has_f = cat in ft.index
    f = ft.loc[cat] if has_f else None
    has_int = has_f and pd.notna(f.lower)
    err = errors.loc[cat] if cat in errors.index else None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Last actual ({fu.LAST_COMPLETE_MONTH:%Y-%m})", fmt(hist.iloc[-1]) if hist.index[-1] == fu.LAST_COMPLETE_MONTH else "n/a")
    if has_f:
        c2.metric(f"Forecast ({fu.FORECAST_MONTH:%Y-%m})", fmt(f.forecast), f"{(f.forecast - f.last_actual) / f.last_actual:+.1%} vs last month")
    else:
        c2.metric("Forecast", "n/a")
    c3.metric(f"{LEVEL} interval", f"{fmt(f.lower)} – {fmt(f.upper)}" if has_int else "not reliable")
    if err is not None:
        c4.metric("Typical error (backtest)", f"±{err.MAPE_pct:.0f}%")

    if not has_f:
        st.warning("No forecast: this category does not have 3 consecutive months of data ending in "
                   f"{fu.LAST_COMPLETE_MONTH:%Y-%m}, which the model needs.")
    elif has_int:
        cov = coverage.set_index("group").loc["large"]
        st.success(f"Reliable range: for large categories the {LEVEL} interval contained the actual value in "
                   f"{cov.actual_coverage:.0%} of backtest months. A prudent planning number is the upper bound "
                   f"(**{fmt(f.upper)}**).")
    else:
        st.warning(f"Low confidence: small category (avg monthly sales ≈ {fmt(hist.mean())}), backtest error ≈ "
                   f"{err.MAPE_pct:.0f}%. No interval is shown because even very wide intervals missed more than half of the "
                   "actual values in the backtest. Use the forecast as a rough hint only.")

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(hist.index, hist.values, "o-", label="Actual", color="#1f4e79")
    bt = test_pred[test_pred.category == cat].sort_values("target_month")
    if len(bt):
        ax.plot(bt.target_month, bt.predicted, "s--", label="Backtest prediction (train-only model)", color="#d98c00")
    if has_f:
        ax.plot([fu.LAST_COMPLETE_MONTH, fu.FORECAST_MONTH], [f.last_actual, f.forecast], ":", color="#c0392b")
        if has_int:
            ax.errorbar([fu.FORECAST_MONTH], [f.forecast], yerr=[[f.forecast - f.lower], [f.upper - f.forecast]],
                        fmt="o", color="#c0392b", capsize=6, markersize=9, label=f"Forecast + {LEVEL} interval", zorder=5)
        else:
            ax.scatter([fu.FORECAST_MONTH], [f.forecast], s=90, color="#c0392b", zorder=5, label="Forecast")
    ax.set_title(cat); ax.set_ylabel("Total sales"); ax.legend(); ax.grid(alpha=.25)
    st.pyplot(fig); plt.close(fig)

    with st.expander("How is this forecast made?"):
        st.write("The model uses this category's sales, orders, discount, late rate, profit and benefit for the last three months, "
                 "plus the 3-month average of sales and the month of the year. The backtest line comes from a separate model trained "
                 "only on data before 2017-03, so it is an honest test. The interval comes from the model's own out-of-sample errors "
                 "in the 6 months before the test period (split-conformal on the log ratio actual/predicted).")

# =========================================================
elif page == "Category segments":
    st.title("Category segments")
    st.caption("KMeans (k=3) on average monthly sales and volatility (coefficient of variation). "
               f"Silhouette score = {info['silhouette']['3']:.2f} (k=2: {info['silhouette']['2']:.2f}, k=4: {info['silhouette']['4']:.2f}).")
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, col in SEG_COLORS.items():
        d = seg[seg.segment == name]
        ax.scatter(d.avg_monthly_sales, d.cv_sales, s=70, color=col, label=f"{name} ({len(d)})")
        for c, r in d.iterrows():
            if name != "Small & volatile" or r.avg_monthly_sales > 3000:
                ax.annotate(c, (r.avg_monthly_sales, r.cv_sales), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.set_xscale("log"); ax.set_xlabel("Average monthly sales (log scale)"); ax.set_ylabel("Volatility (std / mean of monthly sales)")
    ax.legend(); ax.grid(alpha=.25); st.pyplot(fig); plt.close(fig)

    summ = (seg.groupby("segment").agg(categories=("months", "size"), avg_monthly_sales=("avg_monthly_sales", "mean"),
                                       volatility=("cv_sales", "mean"))
            .reindex(list(SEG_COLORS)).round(2))
    st.dataframe(summ, use_container_width=True)
    st.subheader("How to treat each segment (suggested)")
    st.markdown("""
- **Large & stable:** forecast + interval are usable for planning (backtest coverage ≈ 89%). Plan to the upper bound for safety.
- **Mid-size:** forecast is usable but errors are larger; no reliable interval. Review manually.
- **Small & volatile:** not forecastable with this data (errors 40–90%). Better to use simple rules (min stock / reorder point) than the forecast.
    """)
    st.dataframe(seg.reset_index()[["category", "segment", "avg_monthly_sales", "cv_sales", "months"]].round(2),
                 use_container_width=True, hide_index=True)
    st.caption("Profit ratio (≈0.12), late rate (≈0.55) and discount (≈0.10) are almost identical across all categories, "
               "so they do not help to separate segments.")

# =========================================================
elif page == "Category overview":
    st.title("Category overview")
    ov = (data.groupby("category").agg(avg_monthly_sales=("dc_total_sales", "mean"))
          .join(seg[["segment"]]).join(errors[["MAPE_pct"]]).join(ft[["last_actual", "forecast", "lower", "upper"]])
          .sort_values("avg_monthly_sales", ascending=False).reset_index())
    ov["forecast_change_%"] = (ov.forecast / ov.last_actual - 1) * 100
    st.dataframe(ov.round(1).rename(columns={"MAPE_pct": "backtest_error_%", "forecast": f"forecast_{fu.FORECAST_MONTH:%Y-%m}",
                                             "last_actual": f"actual_{fu.LAST_COMPLETE_MONTH:%Y-%m}",
                                             "lower": f"lower_{LEVEL}", "upper": f"upper_{LEVEL}"}),
                 use_container_width=True, hide_index=True)
    top = ov.dropna(subset=["forecast"]).head(10)
    fig, ax = plt.subplots(figsize=(11, 4))
    x = np.arange(len(top))
    ax.bar(x - .2, top.last_actual, width=.4, label="Last actual", color="#1f4e79")
    ax.bar(x + .2, top.forecast, width=.4, label="Forecast", color="#d98c00")
    ax.set_xticks(x); ax.set_xticklabels(top.category, rotation=30, ha="right"); ax.legend()
    ax.set_title("Top categories: last actual vs forecast"); st.pyplot(fig); plt.close(fig)
    st.caption("Intervals exist only for large categories. Small ones had a backtest error of 40–90%.")

# =========================================================
elif page == "Model performance":
    st.title("Model performance")
    st.caption("Test period: 2017-03 → 2017-09 (109 rows). Lower MAE/RMSE is better.")
    sales = load_csv("model_comparison_sales.csv")
    st.subheader("Sales forecast — model comparison")
    st.dataframe(sales, use_container_width=True, hide_index=True)
    base = sales[sales.Model == "Baseline: last month"].MAE.iloc[0]
    best = sales[~sales.Model.str.startswith("Baseline")].iloc[0]
    st.info(f"Best ML model: **{best.Model}** — MAE {best.MAE:,.0f} vs {base:,.0f} for the “last month” baseline "
            f"(≈{(1 - best.MAE / base):.0%} better). Sales are almost flat over time, so the baseline is already strong "
            "and the gain is small and within the noise of a 109-row test.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Actual vs predicted")
        fig, ax = plt.subplots(figsize=(5, 4.5))
        ax.scatter(test_pred.actual, test_pred.predicted, alpha=.6)
        m = max(test_pred.actual.max(), test_pred.predicted.max()) * 1.05
        ax.plot([0, m], [0, m], "r--"); ax.set_xlim(0, m); ax.set_ylim(0, m)
        ax.set_xlabel("Actual"); ax.set_ylabel("Predicted"); st.pyplot(fig); plt.close(fig)
    with col2:
        st.subheader("Feature importance")
        imp = load_csv("feature_importance.csv").head(10).iloc[::-1]
        fig, ax = plt.subplots(figsize=(5, 4.5))
        ax.barh(imp.feature, imp.mae_increase, color="#1f4e79"); ax.set_xlabel("Increase in MAE when shuffled")
        st.pyplot(fig); plt.close(fig)
    st.caption("Permutation importance on the test set. Recent sales dominate; discount, late rate and profit add almost nothing.")

    st.subheader(f"Prediction intervals — {LEVEL} nominal coverage, checked on the test period")
    st.dataframe(coverage.round(3), use_container_width=True, hide_index=True)
    st.info("Large categories: the interval is well calibrated and narrow (≈ ±11%). "
            "Small categories: coverage is far below the target even though the interval is very wide, so no interval is shown for them in the app.")

    st.subheader("Late-rate forecast (negative result)")
    late = load_csv("model_comparison_late_rate.csv")
    st.dataframe(late, use_container_width=True, hide_index=True)
    st.warning("R² is ≤ 0 for every model: no model beats simply predicting the average late rate (~0.55). "
               "Late rate is not predictable from the signals in this file, so the app does not offer a late-rate forecast.")

# =========================================================
else:
    st.title("Data notes — ملاحظات على الداتا")
    st.markdown("""
<div dir="rtl" style="text-align:right">

### إيه اللي اتعمل في الداتا قبل التدريب
- **استبعاد صفوف الـ Logs (30 صف):** فئاتها (apparel وfan shop وgolf...) مختلفة عن فئات DataCo، والـ outer join ملأ مبيعاتها بصفر.
- **استبعاد الشهور بعد 2017-09:** البيانات ناقصة (مثلاً Fishing نزلت من حوالي 490 أوردر لـ 16)، وده انهيار وهمي وليس سلوك سوق.
- **الإبقاء على 24 فئة** ليها 24 شهر أو أكتر من التاريخ.
- **استبعاد `conversion_rate`** (كله صفر) وأعمدة `lg_*` و`high_*` من الـ features.

### حدود الثقة (Prediction Intervals)
- بنحسبها من أخطاء الموديل نفسه في آخر 6 شهور قبل فترة الاختبار (split-conformal).
- **للفئات الكبيرة** (متوسط مبيعات 10 آلاف أو أكتر): الحدود دقيقة (غطّت حوالي 89% من الحالات والمفروض 90%) وضيقة (حوالي ±11%).
- **للفئات الصغيرة:** حتى مع حدود واسعة جداً، غطّت أقل من نص الحالات (حوالي 43%)، فمش بنعرض لها حدود لأنها هتبقى مضللة.

### تقسيم الفئات (Segments)
- KMeans على متوسط المبيعات والتذبذب، بـ 3 مجموعات: كبيرة ومستقرة، متوسطة، وصغيرة ومتذبذبة.
- الخصم ونسبة الربح ونسبة التأخير شبه متطابقين في كل الفئات، فمش بيفرّقوا بين المجموعات.

### ليه مفيش تصنيف (Classification)؟
- `high_interest_low_sales`: كل الحالات الإيجابية (30) هي صفوف الـ Logs، وفترة التدريب مفيهاش ولا حالة.
- `high_late_risk`: قيمته 1 في 95% من الصفوف، والـ 0 أغلبه صفوف الـ Logs (late_rate=0 بسبب `fillna(0)`). أي دقة عالية فيه هي مجرد نسبة الأغلبية.

### حدود التطبيق
- المبيعات شبه ثابتة، فالـ baseline قوي والتحسن من الـ ML صغير.
- اختبار الأداء على 109 صف فقط (2017-03 → 2017-09).
- التوقع للشهر **{month}** مش لـ "الشهر الجاي الحقيقي"، لأن آخر بيانات مكتملة هي 2017-09.
- لازم mapping صحيح بين فئات الـ Logs وDataCo قبل استخدام بيانات الزيارات.

</div>
""".replace("{month}", f"{fu.FORECAST_MONTH:%Y-%m}"), unsafe_allow_html=True)
