# Demand Forecasting and Inventory Optimization

Forecasting daily demand for 100 retail SKUs and turning those forecasts into an inventory policy that is **proven in simulation** to meet a 95% service level target.

Most forecasting projects stop at a model score. This one follows the forecast through to the business decision it supports: how much safety stock each product needs, when to reorder, and whether that policy actually prevents stockouts when replayed against real demand.

## Key results

| Metric | Result |
|---|---|
| Forecast error (WAPE), 7 days ahead | **20.7%**, versus 42.8% for a naive forecast |
| Replenishment cycles without a stockout | **97.1%** (target: 95%) |
| Share of demand served from stock | **99.9%** |
| Uncertainty used for safety stock | **30% lower** than using raw demand variability |

Results are from the held out test period (September 2023 to January 2024), which the model never saw during training or calibration.

## The business problem

A retailer with 5 stores and 20 products needs to decide, every day, which items to reorder and how much buffer stock to hold. Too little stock means lost sales and unhappy customers. Too much ties up cash and shelf space. The goal is to find the smallest amount of stock that still meets a target service level.

## Dataset

`sales_data.csv` contains 76,000 rows: one row per product, per store, per day, covering 760 days (January 2022 to January 2024). Each row includes:

* **Identity:** store, product, category, region
* **Commercial levers:** price, discount, promotion, competitor pricing
* **External conditions:** weather, season, epidemic flag
* **Outcomes:** inventory level, units sold, units ordered, and **demand** (the prediction target)

The data is complete, with no missing values or duplicate records.

## Approach

### 1. Exploratory analysis
Demand was examined over time and across categories, regions, weather, promotions and epidemic periods to understand what moves it before any modeling.

### 2. Feature engineering
* **Calendar features** (month, day of week, day of year, week of year) capture weekly and seasonal patterns.
* **Lag features** give the model each SKU's recent demand history: demand 7, 8, 14, 21 and 28 days earlier, plus 7 and 28 day rolling averages.
* Every lag is shifted by the **7 day forecast horizon**, so the model only uses information that would genuinely be available a week in advance, matching the replenishment lead time.
* Categorical columns are one hot encoded.

### 3. Preventing data leakage
`Units Sold` and `Units Ordered` are only known after the day ends, so they are excluded from the inputs. `Inventory Level` was also tested for leakage by comparing models with and without it; removing it made no difference to accuracy, confirming the results are not inflated.

### 4. Time based validation
The timeline is split chronologically into three periods, never randomly:

| Period | Dates | Purpose |
|---|---|---|
| Training | Feb 2022 to Jun 2023 | Fit the model |
| Calibration | Jun 2023 to Sep 2023 | Measure forecast error for safety stock |
| Test | Sep 2023 to Jan 2024 | Final accuracy and policy evaluation |

Keeping calibration separate from test means the inventory policy is judged on days it was not tuned on.

### 5. Forecasting model and baselines
An XGBoost regressor is compared against three references to show what the model actually adds:

| Model | WAPE | R² |
|---|---|---|
| Naive (demand 7 days earlier) | 42.8% | below 0 |
| Rolling 28 day average | 35.2% | 0.01 |
| XGBoost without lag features | 20.4% | 0.62 |
| **XGBoost with lag features** | **20.7%** | **0.61** |

WAPE is total absolute error as a share of total demand (lower is better).

### 6. Inventory policy per SKU
Safety stock and reorder points are calculated separately for each of the 100 store and product combinations, using three principles:

* **Forecast error, not raw demand, drives safety stock.** Once replenishment is planned from a forecast, stockout risk comes from the forecast being wrong. Median daily uncertainty falls from 39.5 units (raw demand) to 27.7 units (forecast error).
* **The protection interval includes the review period.** Stock is reviewed daily, so each reorder decision must cover the 7 day lead time plus 1 review day. Ignoring the review period is a common reason real reorder policies miss their service level.
* **Error over the full interval is measured directly** by summing forecast errors over rolling 8 day windows, rather than relying only on the textbook square root shortcut, which assumes errors are independent from day to day.

```
Safety stock   = z × σ(forecast error over the protection interval)
Reorder point  = forecast demand over the protection interval + safety stock
z              = 1.645 for a 95% service level
```

### 7. Replenishment simulation
The policy is replayed day by day across the five month test period for every SKU:

1. Orders due that day arrive at the start of the day.
2. Actual demand is served from stock; any shortfall is a lost sale.
3. If stock on hand plus stock on order falls to the reorder point, an order is placed, arriving 7 days later.

The reorder point is dynamic, using the forecast for the specific upcoming days rather than a flat average.

**Outcome:** 97.1% of replenishment cycles completed without a stockout, against a 95% target, with 99.9% of all demand served from stock.

## Findings

* **The inventory policy works as designed.** The simulation confirms the service level target is met across the portfolio of SKUs, which is the strongest evidence the approach would hold up in practice.
* **Lag features added little on this dataset.** Demand here depends only weakly on its own recent history (correlation of about 0.17 with the same SKU one week earlier), which is unusual for retail. Accuracy comes mainly from price, category, promotion and epidemic effects.
* **The model learned the dataset's price and category patterns.** Within each category, higher prices appear alongside higher demand, which suggests prices in this dataset were set with knowledge of demand. This is a reminder to validate relationships before treating them as causal.
* **Daily forecast errors were close to independent**, so the direct measurement of lead time error (84 units) and the square root shortcut (78 units) agreed closely here. The notebook measures it directly because real world errors often persist, and the shortcut then understates risk.

## Limitations

* Lead time (7 days) and service level (95%) are configurable assumptions, not values taken from the data.
* Weather is used as if known in advance; in production a weather forecast would be used instead.
* The simulation treats unmet demand as lost and ignores minimum order quantities, costs and shelf capacity.
* With about 10 replenishment cycles per SKU in the test period, individual SKU service levels are noisy; the aggregate figure is the more reliable measure.

## Future work: scaling to production

The notebook forms the AI layer of a larger system. A production version would add:

* **Cloud data layer:** a warehouse (for example BigQuery) holding SKU history, fed by point of sale, warehouse and ERP systems.
* **Scheduled ingestion:** daily automated loads replacing the manual CSV upload.
* **Pipeline automation:** scheduled retraining, daily forecasts, and automatic reorder flags written back to the warehouse.
* **Analytics layer:** a Power BI dashboard showing forecast versus actual, stock versus reorder point, and SKUs at risk.
* **Real supplier lead times** with their own variability built into safety stock, plus monitoring that alerts when accuracy or service level drifts.

## How to run

**Google Colab (easiest)**
1. Open `demand_forecast.ipynb` in Colab.
2. Upload `sales_data.csv` when prompted, or place it at `/content/sales_data.csv`.
3. Select **Runtime > Run all**.

**Locally**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy xgboost jupyter
jupyter notebook demand_forecast.ipynb
```
Set `DATA_PATH` in the configuration cell to the location of `sales_data.csv`.

All assumptions (forecast horizon, lead time, service level, split sizes) are set in a single configuration cell at the top of the notebook.

## Repository structure

```
├── demand_forecast.ipynb   # Full analysis: EDA, modeling, inventory policy, simulation
├── sales_data.csv          # Dataset
└── README.md
```

## Tools

Python, pandas, NumPy, XGBoost, scikit-learn, SciPy, Matplotlib, Seaborn, Google Colab

## Author

**Durga Sampath Kumar Tandle**
MS in Management of Technology, Arizona State University
[LinkedIn](https://www.linkedin.com/in/sampath-kumar-tandle-85768619a/)
