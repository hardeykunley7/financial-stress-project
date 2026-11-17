# Financial Stress Project - Predicting Liquidity Stress Next 30 Days
# Author: Samuel Joseph Adekunle
# Train: Train_1_-4.csv (40,000 rows) | Test: Test-3.csv (30,000 rows)
# Target: liquidity_stress_next_30d

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score, precision_score, recall_score
import lightgbm as lgb

TARGET = 'liquidity_stress_next_30d'
BAL = [f'balance_{m}' for m in ['M1','M2','M3','M4','M5','M6']]
CHANNELS = ['paybill','merchantpay','transfer_from_bank','mm_send','received','deposit','withdraw']

def add_features(df):
    df = df.copy()
    df['bal_recent3'] = df[[f'balance_M{i}' for i in [1,2,3]]].mean(axis=1)
    df['bal_old3'] = df[[f'balance_M{i}' for i in [4,5,6]]].mean(axis=1)
    df['bal_drop'] = df['bal_old3'] - df['bal_recent3']
    df['bal_drop_ratio'] = df['bal_drop'] / (df['bal_old3'] + 1)
    df['bal_cv'] = df[BAL].std(axis=1) / (df[BAL].mean(axis=1) + 1)
    df['bal_min'] = df[BAL].min(axis=1)
    df['bal_max'] = df[BAL].max(axis=1)
    df['bal_mean'] = df[BAL].mean(axis=1)
    df['bal_m1_minus_m4'] = df['balance_M1'] - df['balance_M4']
    for ch in CHANNELS:
        vol_cols = [f'{ch}_volume_M{i}' for i in range(1,7)]
        val_cols = [f'{ch}_value_M{i}' for i in range(1,7)]
        vol_cols = [c for c in vol_cols if c in df.columns]
        val_cols = [c for c in val_cols if c in df.columns]
        if vol_cols:
            df[f'{ch}_vol_6m'] = np.log1p(df[vol_cols].sum(axis=1))
            df[f'{ch}_vol_recent'] = np.log1p(df[[c for c in vol_cols if 'M1' in c or 'M2' in c or 'M3' in c]].sum(axis=1))
            df[f'{ch}_vol_old'] = np.log1p(df[[c for c in vol_cols if 'M4' in c or 'M5' in c or 'M6' in c]].sum(axis=1))
            df[f'{ch}_vol_drop'] = df[f'{ch}_vol_old'] - df[f'{ch}_vol_recent']
        if val_cols:
            df[f'{ch}_val_6m'] = np.log1p(df[val_cols].sum(axis=1))
    if 'arpu' in df.columns:
        df['log_arpu'] = np.log1p(df['arpu'])
    if 'age' in df.columns:
        df['age_10'] = df['age'] / 10
    if 'activity' in df.columns:
        df['activity_low'] = (df['activity'] <= 0.3).astype(int)
    for col in ['gender','smartphone','segment','earning_pattern','region']:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True, dtype=float)
            df = pd.concat([df, dummies], axis=1)
    return df

def main():
    train = pd.read_csv('Train_1_-4.csv')
    test = pd.read_csv('Test-3.csv')
    print(f"Train: {train.shape}, Test: {test.shape}")
    print(f"Target rate: {train[TARGET].mean():.3f}")
    train = add_features(train)
    test = add_features(test)
    exclude = [TARGET, 'ID', 'id', 'customer_id'] + BAL
    exclude += [c for c in train.columns if c in ['gender','smartphone','segment','earning_pattern','region','arpu','age','activity']]
    exclude += [c for c in train.columns if any(x in c for x in CHANNELS) and ('_M' in c)]
    features = [c for c in train.columns if c not in exclude and pd.api.types.is_numeric_dtype(train[c])]
    print(f"Features ({len(features)}): {features}")
    X = train[features].fillna(0)
    y = train[TARGET]
    X_test = test[features].fillna(0)
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    sc = StandardScaler()
    lr = LogisticRegression(max_iter=1000)
    lr.fit(sc.fit_transform(Xtr), ytr)
    p_va = lr.predict_proba(sc.transform(Xva))[:,1]
    print(f"Logistic Val ROC-AUC: {roc_auc_score(yva, p_va):.3f} LogLoss: {log_loss(yva, p_va):.3f}")
    pred = (p_va >= 0.5).astype(int)
    print(f"Acc: {accuracy_score(yva, pred):.3f} Prec: {precision_score(yva, pred):.3f} Rec: {recall_score(yva, pred):.3f}")
    k = int(len(yva)*0.10)
    idx = np.argsort(p_va)[::-1][:k]
    print(f"Top10% Precision: {yva.iloc[idx].mean():.3f} Capture: {yva.iloc[idx].sum()/yva.sum():.3f}")
    model = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1)
    model.fit(Xtr, ytr)
    p_va2 = model.predict_proba(Xva)[:,1]
    print(f"LightGBM Val ROC-AUC: {roc_auc_score(yva, p_va2):.3f} LogLoss: {log_loss(yva, p_va2):.3f}")
    model.fit(X, y)
    test_pred = model.predict_proba(X_test)[:,1]
    out = pd.DataFrame({'ID': test['ID'] if 'ID' in test.columns else test.index, TARGET: test_pred})
    out.to_csv('submission.csv', index=False)
    print("Saved submission.csv", out.shape)

if __name__ == '__main__':
    main()
