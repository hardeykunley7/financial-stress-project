# Financial Stress Project — Predicting Liquidity Stress Next 30 Days

**Name:** Samuel Joseph Adekunle
**Project:** Financial Stress Project
**Course:** Data Science and Machine Learning Project

## Task
Predict `liquidity_stress_next_30d` (1 = liquidity stress, 0 = no stress) for 30,000 customers using 6 months of mobile-money behaviour. M1 = most recent month, M6 = oldest month.

## Files Used
- `Train_1_-4.csv` — 40,000 rows x 184 cols (Train)
- `Test-3.csv` — 30,000 rows x 183 cols (Test, no target)
- `data_dictionary-1.csv` — column definitions
- `Project_Description-3.docx` — project instructions

No other files were used. No Climate files were mixed in.

## Data
- Train: 40,000 rows
- Test: 30,000 rows
- Missing values: 0 in Train, 0 in Test
- Target balance: 15.0% stressed / 85.0% not stressed — imbalanced

## Key Finding
Balance trajectory is the signal, not demographics.

Non-stressed customers are flat. Stressed customers had much higher balances in M4-M6 (~20,100-20,700) then collapsed to ~14,000 in M1-M3.

- No stress (0): M1 14,799, M2 14,706, M3 14,764, M4 13,750, M5 13,808, M6 13,623
- Stress (1): M1 14,035, M2 14,069, M3 13,902, M4 20,163, M5 20,489, M6 20,694

Other findings:
- Segment: HVC 19.0% stressed, LVC 17.2%, MVC 12.6%
- Gender / Region / Smartphone / Earning pattern: almost flat (13.4%-15.5%)
- Activity: U-shaped. Activity <=0.3: 18.2% stressed, 0.5-0.7: 13.1%
- Age: flat. ARPU: U-shaped.

## Features (60 features)
- Balance: bal_recent3 = mean(M1-M3), bal_old3 = mean(M4-M6), bal_drop, bal_drop_ratio, bal_cv, bal_min/max/mean, bal_m1_minus_m4
- Per channel (paybill, merchantpay, transfer_from_bank, mm_send, received, deposit, withdraw): 6-month log1p(volume), log1p(total_value), recency_ratio, drop
- Profile: log1p(arpu), age/10, activity, activity_low
- One-hot: gender, smartphone, segment, earning_pattern, region
- No leakage: all features use only M1-M6 history, target is next 30 days.

## Model & Results (Stratified hold-out: 32,000 train / 8,000 validation)
Baseline: Logistic Regression (standardised, L2)

- ROC-AUC: Train 0.809 / Validation 0.805
- Log Loss: Train 0.335 / Validation 0.338
- Accuracy @0.5: Train 0.866 / Validation 0.867
- Precision @0.5: Train 0.687 / Validation 0.688
- Recall @0.5: Train 0.199 / Validation 0.206
- Precision in top 10% scored: 0.554 / 0.559
- Capture in top 10% scored: 36.9% / 37.3%

Recommended final model: LightGBM / HistGradientBoosting on the same 60 features, compared on ROC-AUC + PR-AUC + top-decile lift. Logistic is kept as interpretable baseline.

## Recommendation
Do not use 0.5 threshold for operations. Use ranking: contacting the top 10% scored captures ~37% of all future stress cases at 56% precision, 3.7x base rate. Prioritise HVC + LVC + low-activity + balance-drop customers.

## How to Run
pip install -r requirements.txt
jupyter notebook Financial_Stress_Notebook.ipynb

## Submission
submission.csv has columns: ID, liquidity_stress_next_30d with probabilities.
