import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
     
df = pd.read_csv('http://logopt.com/data/peyton_manning.csv')

from sktime.forecasting.model_selection import temporal_train_test_split

TIME_SPAN = 90

train, test = temporal_train_test_split(df, test_size=TIME_SPAN)

from merlion.utils import TimeSeries
from merlion.models.defaults import DefaultForecasterConfig, DefaultForecaster
# 参考 datasets の使用
import sys
sys.path.append("/content/Merlion/ts_datasets")

from merlion.utils import TimeSeries
from ts_datasets.forecast import M4

# Data loader returns pandas DataFrames, which we convert to Merlion TimeSeries
time_series, metadata = M4(subset="Hourly")[0]

model = DefaultForecaster(DefaultForecasterConfig())

# データ変換
train_ts = TimeSeries.from_pd(train.set_index("ds"))
test_ts = TimeSeries.from_pd(test.set_index("ds"))

# 学習
model.train(train_data=train_ts)

# 予測データの作成
pred_merlion, test_err_merlion = model.forecast(time_stamps=test_ts.time_stamps)