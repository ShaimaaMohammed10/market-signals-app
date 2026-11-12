# Market Signals — ML Project

توقع مبيعات الشهر القادم لكل Category من `market_signals.csv`، مع حدود ثقة للفئات الكبيرة وتقسيم الفئات لمجموعات، وتطبيق Streamlit لعرض النتايج.

## الملفات
| الملف | الوصف |
|---|---|
| `Market_Signals_ML_fixed.ipynb` | الـ notebook الكامل: Cleaning → Features → Models → Tuning → Importance → **Prediction intervals** → **Segments** → Save |
| `app.py` | تطبيق Streamlit (5 صفحات: Forecast / Category segments / Category overview / Model performance / Data notes) |
| `forecast_utils.py` | دوال تنضيف الداتا والـ features |
| `prepare_app_assets.py` | بيبني كل ملفات `outputs/` (النتايج، الحدود، المجموعات، الموديل النهائي) |
| `market_signals.csv` | الداتا |
| `README_data_market_signals.md` | وصف أعمدة الداتا |
| `outputs/*.csv, *.json` | ملفات جاهزة بيقراها التطبيق (التوقعات، حدود الثقة، المجموعات، مقارنة الموديلات) |
| `requirements.txt` | مكتبات التطبيق فقط (pandas, numpy, matplotlib, streamlit) |
| `requirements-dev.txt` | مكتبات الـ notebook والسكريبت (بتضيف scikit-learn و joblib) |

## تشغيل التطبيق
```
pip install -r requirements.txt
streamlit run app.py
```
التطبيق بيقرا ملفات `outputs/` الجاهزة وما بيحتاجش scikit-learn ولا ملف الموديل.

## إعادة بناء النتايج والموديل
```
pip install -r requirements-dev.txt
python prepare_app_assets.py
```
بيطلّع `outputs/app_model.joblib` (الموديل النهائي) وباقي الملفات. الموديل مش مرفق في الـ zip (حجمه ~25MB).

## استخدام الموديل (مثلاً للـ AI Agent)
```python
import joblib
bundle = joblib.load("outputs/app_model.joblib")
pred = bundle["model"].predict(new_df[bundle["feature_columns"]])
```
أو اقرأي `outputs/forecast_table.csv` مباشرة: فيه لكل فئة التوقع وحدّه الأدنى والأعلى (للفئات الكبيرة).

## النتايج (Test: 2017-03 → 2017-09، 109 صف)
- **المبيعات:** أحسن موديل Random Forest (tuned) بـ MAE حوالي 3.6 ألف مقابل 3816 للـ baseline "الشهر اللي فات" (تحسن حوالي 5% وفي حدود الضوضاء، لأن المبيعات شبه ثابتة).
- **حدود الثقة 90%:**
  - فئات كبيرة (متوسط مبيعات ≥ 10 آلاف): غطّت **89%** من الحالات بعرض ≈ ±11%.
  - فئات صغيرة: غطّت **43%** فقط حتى مع حدود واسعة جداً، فمش بتتعرض.
- **المجموعات (KMeans, silhouette ≈ 0.70):** Large & stable (9 فئات) / Mid-size (7) / Small & volatile (8).
- **التأخير:** R² سالب، يعني مش قابل للتنبؤ بالمتغيرات المتاحة.

## قرارات تنضيف الداتا
- استبعاد صفوف الـ Logs (30 صف): مبيعاتها صفر بحكم الـ outer join ولا تتطابق مع فئات DataCo.
- استبعاد الشهور بعد 2017-09: بيانات ناقصة (مثلاً Fishing نزلت من ~490 أوردر لـ 16).
- الإبقاء على الفئات اللي ليها 24 شهر أو أكتر (24 فئة).
- استبعاد `conversion_rate` (كله صفر) وأعمدة `lg_*` و`high_*`.
- حذف مهام التصنيف (`high_late_risk`, `high_interest_low_sales`): الـ labels ناتجة عن `fillna(0)` بعد الـ outer join.

## حدود المشروع
- المبيعات شبه ثابتة، فالـ baseline قوي والتحسين من الـ ML صغير.
- الاختبار على 109 صف فقط؛ وحدود الثقة معايرة على 144 مشاهدة، فالأرقام تقريبية.
- التوقع لشهر 2017-10 لأن آخر بيانات مكتملة هي 2017-09، وليس "الشهر الجاي الحقيقي".
- الفئات الصغيرة والمتذبذبة: التوقع إرشادي فقط.
- بيانات الـ Logs مش مستخدمة: محتاجة mapping صحيح لفئات DataCo. ولربط الشغل بـ `demand_shortage_features.csv` محتاجين mapping SKU → Category.
