# %%
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold

train = pd.read_csv('input/sample-data/train_preprocessed.csv')
train_x = train.drop(['target'], axis=1)
train_y = train['target']
test_x = pd.read_csv('input/sample-data/test_preprocessed.csv')

kf = KFold(n_splits=4, shuffle=True, random_state=71)
tr_idx, va_idx = list(kf.split(train_x))[0]
tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

dtrain = xgb.DMatrix(tr_x, label=tr_y)
dvalid = xgb.DMatrix(va_x, label=va_y)
dtest = xgb.DMatrix(test_x)
# %%
params = {'objective': 'binary:logistic', 'silent': 1, 'random_state': 71}
num_round = 50

watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
model = xgb.train(params, dtrain, num_round, evals=watchlist)

va_pred = model.predict(dvalid)
score = log_loss(va_y, va_pred)
print(f'logloss: {score:.4f}')
pred = model.predict(dtest)

print(pred)

# %%
print(len(pred))

# %%
params = {
    'objective': 'binary:logistic',
    'silent': 1,
    'random_state': 71,
    'eval_metric': 'logloss'
}
num_round = 500
watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
model = xgb.train(params,
                  dtrain,
                  num_round,
                  evals=watchlist,
                  early_stopping_rounds=20)

pred = model.predict(dtest, ntree_limit=model.best_ntree_limit)

# %%
