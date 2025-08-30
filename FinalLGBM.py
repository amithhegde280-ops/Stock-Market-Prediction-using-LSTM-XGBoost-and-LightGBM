import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import train_test_split


data = pd.read_csv('GOOGL.csv')


prices = data['Close'].values


df = pd.DataFrame(prices, columns=['Close'])


df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
df = df.dropna()  


threshold = 2  
mean = df['Log_Returns'].mean()
std = df['Log_Returns'].std()
df['Z_Score'] = (df['Log_Returns'] - mean) / std
outliers = df[np.abs(df['Z_Score']) > threshold]


def create_lagged_features(df, n_lags):
    df_lagged = pd.concat([df['Log_Returns'].shift(i) for i in range(n_lags, 0, -1)], axis=1)
    df_lagged.columns = [f't-{i}' for i in range(n_lags, 0, -1)]
    df_lagged['t'] = df['Log_Returns']
    df_lagged.dropna(inplace=True)
    return df_lagged

n_lags = 2
df_lagged_outliers = create_lagged_features(outliers, n_lags)


scaler = MinMaxScaler()
scaled_data_outliers = scaler.fit_transform(df_lagged_outliers)


X_outliers = scaled_data_outliers[:, :-1]
y_outliers = scaled_data_outliers[:, -1]


X_outliers = X_outliers.reshape((X_outliers.shape[0], X_outliers.shape[1], 1))


train_size = int(len(X_outliers) * 0.8)
X_train_outliers, X_test_outliers = X_outliers[:train_size], X_outliers[train_size:]
y_train_outliers, y_test_outliers = y_outliers[:train_size], y_outliers[train_size:]


model_lstm_outliers = Sequential([
    LSTM(units=100, return_sequences=True, input_shape=(n_lags, 1)),
    LSTM(units=100, return_sequences=False),
    Dense(units=1)
])

model_lstm_outliers.compile(optimizer='adam', loss='mean_squared_error')
model_lstm_outliers.fit(X_train_outliers, y_train_outliers, epochs=20, batch_size=32, validation_split=0.2, verbose=1)


train_predict_lstm_outliers = model_lstm_outliers.predict(X_train_outliers)
test_predict_lstm_outliers = model_lstm_outliers.predict(X_test_outliers)


train_predict_lstm_outliers = train_predict_lstm_outliers.flatten()
test_predict_lstm_outliers = test_predict_lstm_outliers.flatten()


df_lagged_whole = create_lagged_features(df, n_lags)


scaled_data_whole = scaler.fit_transform(df_lagged_whole)


X_whole = scaled_data_whole[:, :-1]
y_whole = scaled_data_whole[:, -1]


lstm_predictions_for_whole = np.concatenate((train_predict_lstm_outliers, np.full(X_whole.shape[0] - len(train_predict_lstm_outliers), np.nan)))


X_whole_with_lstm = np.column_stack((X_whole, lstm_predictions_for_whole))


X_train_whole, X_test_whole, y_train_whole, y_test_whole = train_test_split(X_whole_with_lstm, y_whole, test_size=0.2, random_state=42)


model_lgbm_whole = lgb.LGBMRegressor(boosting_type='gbdt')
model_lgbm_whole.fit(X_train_whole, y_train_whole)


train_predict_lgbm_whole = model_lgbm_whole.predict(X_train_whole)
test_predict_lgbm_whole = model_lgbm_whole.predict(X_test_whole)


train_rmse_lgbm_whole = np.sqrt(mean_squared_error(y_train_whole, train_predict_lgbm_whole))
test_rmse_lgbm_whole = np.sqrt(mean_squared_error(y_test_whole, test_predict_lgbm_whole))

print(f'LGBM Train RMSE (Whole Dataset): {train_rmse_lgbm_whole:.4f}')
print(f'LGBM Test RMSE (Whole Dataset): {test_rmse_lgbm_whole:.4f}')


plt.figure(figsize=(14, 7))
plt.plot(range(len(y_train_whole)), y_train_whole, label='Actual Train Log Returns (Whole Dataset)')
plt.plot(range(len(y_train_whole)), train_predict_lgbm_whole, label='Predicted Train Log Returns (Whole Dataset)')
plt.plot(range(len(y_train_whole), len(y_train_whole) + len(y_test_whole)), y_test_whole, label='Actual Test Log Returns (Whole Dataset)')
plt.plot(range(len(y_train_whole), len(y_train_whole) + len(y_test_whole)), test_predict_lgbm_whole, label='Predicted Test Log Returns (Whole Dataset)')
plt.title(f'LGBM with LSTM Features - Whole Dataset\nTrain RMSE: {train_rmse_lgbm_whole:.4f}, Test RMSE: {test_rmse_lgbm_whole:.4f}')
plt.xlabel('Days')
plt.ylabel('Log Returns')
plt.legend()
plt.tight_layout()
plt.show()
