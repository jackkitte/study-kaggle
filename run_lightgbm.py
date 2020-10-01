# %%
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss

train = pd.read_csv('input/sample-data/train_preprocessed.csv')
train_x = train.drop(['target'], axis=1)
train_y = train['target']
test_x = pd.read_csv('input/sample-data/test_preprocessed.csv')

kf = KFold(n_splits=4, shuffle=True, random_state=71)
tr_idx, va_idx = list(kf.split(train_x))[0]
tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

# %%
lgb_train = lgb.Dataset(tr_x, tr_y)
lgb_eval = lgb.Dataset(va_x, va_y)

params = {
    'objective': 'binary',
    'seed': 71,
    'verbose': 0,
    'metrics': 'binary_logloss'
}
num_round = 100
categorical_features = ['product', 'medical_info_b2', 'medical_info_b3']
model = lgb.train(params,
                  lgb_train,
                  num_boost_round=num_round,
                  categorical_feature=categorical_features,
                  valid_names=['train', 'valid'],
                  valid_sets=[lgb_train, lgb_eval])

va_pred = model.predict(va_x)
score = log_loss(va_y, va_pred)
print(f'logloss: {score:.4f}')

pred = model.predict(test_x)

# %%
