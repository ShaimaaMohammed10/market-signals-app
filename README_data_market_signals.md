# market_signals.csv

## Overview
This dataset provides a monthly aggregated summary of market signals per product category.
It combines sales/order data (DataCo) with website access logs to give a unified view
of demand, interest, and delivery performance.
Each row represents a unique combination of category, year, and month.

## Purpose
- Analyze market interest vs. actual sales per category.
- Detect categories with high interest but low sales (possible pricing or listing issues).
- Monitor late delivery rates and profitability at the category level.
- Provide market context to an AI agent making supply chain decisions.

## Sources
- DataCo Supply Chain dataset (DataCoSupplyChain_Cleaned.csv) -> sales, orders, discounts, late delivery, profit.
- Access Logs dataset (tokenized_access_logs_Cleaned.csv) -> product views, unique visitors.
Both sources are aggregated by category, year, and month and merged using an outer join.

## Granularity
- Level: Category x Month
- Rows: one per (category, year, month)
- Time span: depends on the source data

## Columns

| Column | Type | Source | Description |
|---|---|---|---|
| category | string | DataCo / Logs | Product category (e.g., Fitness, Golf, Footwear) |
| year | int | Both | Year of the record |
| month | int | Both | Month of the record (1-12) |
| year_month | string | Derived | Combined label in YYYY-MM format |
| dc_total_sales | float | DataCo | Total sales amount for the category |
| dc_total_orders | int | DataCo | Number of unique orders |
| dc_avg_discount | float | DataCo | Average discount rate applied |
| dc_late_rate | float | DataCo | Proportion of late deliveries (0-1) |
| dc_avg_profit_ratio | float | DataCo | Average profit ratio per order item |
| dc_total_benefit | float | DataCo | Total benefit (profit) for the category |
| lg_total_views | int | Access Logs | Total product views in the category |
| lg_unique_visitors | int | Access Logs | Number of unique visitors (IPs) |
| lg_unique_products | int | Access Logs | Number of distinct products viewed |
| conversion_rate | float | Derived | Orders / Views (0 if no views) |
| high_interest_low_sales | int (0/1) | Derived | 1 if views > mean AND sales < mean |
| high_late_risk | int (0/1) | Derived | 1 if dc_late_rate > 0.3 |

## Derived Metrics
- conversion_rate: measures how effectively views turn into orders.
  A low value may indicate pricing, UX, or stock issues.
- high_interest_low_sales: flags categories with high traffic but weak sales,
  a strong signal for business investigation.
- high_late_risk: flags categories with a late delivery rate above 30%.

## Notes & Caveats
- Category names may not fully overlap between DataCo (Category Name) and Access Logs (Department).
  An outer join is used, so some rows may have data from only one source (zeros in the other).
- Missing values in numeric columns are filled with 0 after the outer join.
- This dataset should NOT be merged with demand_shortage_features.csv at the row level.
  They operate at different granularities (Category vs. SKU) and merging them would create
  a cartesian product.
- Category-level signals are meant to complement SKU-level data, not replace it.

## Suggested Use Cases
1. Market Analysis - identify top-performing and underperforming categories.
2. Conversion Funnel - spot categories with high interest but low conversion.
3. Delivery Performance - track late delivery rates per category.
4. Agent Context - provide market context to an AI agent deciding on replenishment.
5. Business Reporting - build dashboards (Streamlit / Power BI) for category performance.

## Example Row
category,year,month,year_month,dc_total_sales,dc_total_orders,dc_avg_discount,dc_late_rate,dc_avg_profit_ratio,dc_total_benefit,lg_total_views,lg_unique_visitors,lg_unique_products,conversion_rate,high_interest_low_sales,high_late_risk
Fitness,2017,9,2017-09,120000.0,450,0.12,0.35,0.28,33600.0,45000,12000,85,0.01,0,1

## Related Files
- demand_shortage_features.csv - SKU-level inventory and demand summary.
- DataCoSupplyChain_Cleaned.csv - raw orders data (source).
- tokenized_access_logs_Cleaned.csv - raw access logs (source).