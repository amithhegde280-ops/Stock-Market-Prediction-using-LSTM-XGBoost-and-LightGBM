import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split

# Load the data
data = pd.read_csv('GOOGL.csv')
prices = data['Close'].values
df = pd.DataFrame(prices, columns=['Close'])

# Calculate log returns
df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
df = df.dropna()

# Define outliers using Z-score
threshold = 2
mean = df['Log_Returns'].mean()
std = df['Log_Returns'].std()
df['Z_Score'] = (df['Log_Returns'] - mean) / std
outliers = df[np.abs(df['Z_Score']) > threshold]

# Create lagged features for the dataset
def create_lagged_features(df, n_lags):
    df_lagged = pd.concat([df['Log_Returns'].shift(i) for i in range(n_lags, 0, -1)], axis=1)
    df_lagged.columns = [f't-{i}' for i in range(n_lags, 0, -1)]
    df_lagged['t'] = df['Log_Returns']
    df_lagged.dropna(inplace=True)
    return df_lagged

n_lags = 2
df_lagged = create_lagged_features(df, n_lags)

# Scale the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df_lagged)

# Split into features (X) and target (y)
X_whole = scaled_data[:, :-1]
y_whole = scaled_data[:, -1]

# Create lagged features for outliers
df_lagged_outliers = create_lagged_features(outliers, n_lags)
scaled_data_outliers = scaler.fit_transform(df_lagged_outliers)

# Split into features (X) and target (y) for outliers
X_outliers = scaled_data_outliers[:, :-1]
y_outliers = scaled_data_outliers[:, -1]
X_outliers = X_outliers.reshape((X_outliers.shape[0], X_outliers.shape[1], 1))

# Train-test split for outliers
train_size = int(len(X_outliers) * 0.8)
X_train_outliers, X_test_outliers = X_outliers[:train_size], X_outliers[train_size:]
y_train_outliers, y_test_outliers = y_outliers[:train_size], y_outliers[train_size:]

# Define and train the LSTM model on outliers
model_lstm_outliers = Sequential([
    LSTM(units=100, return_sequences=True, input_shape=(n_lags, 1)),
    LSTM(units=100, return_sequences=False),
    Dense(units=1)
])

model_lstm_outliers.compile(optimizer='adam', loss='mean_squared_error')
model_lstm_outliers.fit(X_train_outliers, y_train_outliers, epochs=20, batch_size=32, validation_split=0.2, verbose=1)

# Make predictions with the LSTM model on the entire dataset
full_X = scaled_data[:, :-1].reshape(-1, n_lags, 1)
predictions_lstm = model_lstm_outliers.predict(full_X).flatten()

# Combine the original features with LSTM predictions
X_combined = np.concatenate((X_whole, predictions_lstm.reshape(-1, 1)), axis=1)

# Train-test split for the whole dataset
X_train_whole, X_test_whole, y_train_whole, y_test_whole = train_test_split(X_combined, y_whole, test_size=0.2, random_state=42)

# Initialize and train the LightGBM model on the whole dataset
model_lgbm_whole = lgb.LGBMRegressor(boosting_type='gbdt')
model_lgbm_whole.fit(X_train_whole, y_train_whole)

# Initialize and train the XGBoost model on the whole dataset
model_xgb_whole = xgb.XGBRegressor()
model_xgb_whole.fit(X_train_whole, y_train_whole)

# Make predictions with both models on the whole dataset
train_predict_lgbm_whole = model_lgbm_whole.predict(X_train_whole)
test_predict_lgbm_whole = model_lgbm_whole.predict(X_test_whole)

train_predict_xgb_whole = model_xgb_whole.predict(X_train_whole)
test_predict_xgb_whole = model_xgb_whole.predict(X_test_whole)

# Combine predictions using averaging
train_predict_combined = (train_predict_lgbm_whole + train_predict_xgb_whole) / 2
test_predict_combined = (test_predict_lgbm_whole + test_predict_xgb_whole) / 2

# Calculate RMSE for combined predictions
train_rmse_combined = np.sqrt(mean_squared_error(y_train_whole, train_predict_combined))
test_rmse_combined = np.sqrt(mean_squared_error(y_test_whole, test_predict_combined))

print(f'Combined Train RMSE: {train_rmse_combined}')
print(f'Combined Test RMSE: {test_rmse_combined}')

# Plotting
plt.figure(figsize=(14, 7))
plt.plot(range(len(y_train_whole)), y_train_whole, label='Actual Train Log Returns (Whole Dataset)')
plt.plot(range(len(y_train_whole)), train_predict_combined, label='Predicted Train Log Returns (Combined)')
plt.plot(range(len(y_train_whole), len(y_train_whole) + len(y_test_whole)), y_test_whole, label='Actual Test Log Returns (Whole Dataset)')
plt.plot(range(len(y_train_whole), len(y_train_whole) + len(y_test_whole)), test_predict_combined, label='Predicted Test Log Returns (Combined)')
plt.title(f'Combined Model with LSTM, XGBoost, and LightGBM\nTrain RMSE: {train_rmse_combined:.4f}, Test RMSE: {test_rmse_combined:.4f}')
plt.xlabel('Days')
plt.ylabel('Log Returns')
plt.legend()
plt.tight_layout()
plt.show()

# Calculate RMSE for individual models
train_rmse_lgbm = np.sqrt(mean_squared_error(y_train_whole, train_predict_lgbm_whole))
test_rmse_lgbm = np.sqrt(mean_squared_error(y_test_whole, test_predict_lgbm_whole))

train_rmse_xgb = np.sqrt(mean_squared_error(y_train_whole, train_predict_xgb_whole))
test_rmse_xgb = np.sqrt(mean_squared_error(y_test_whole, test_predict_xgb_whole))

# Print RMSE for individual models
print(f'LightGBM Train RMSE: {train_rmse_lgbm}')
print(f'LightGBM Test RMSE: {test_rmse_lgbm}')
print(f'XGBoost Train RMSE: {train_rmse_xgb}')
print(f'XGBoost Test RMSE: {test_rmse_xgb}')

# Make predictions with the LSTM model on the outliers dataset
predictions_lstm_outliers = model_lstm_outliers.predict(X_outliers).flatten()

# Calculate RMSE for LSTM on outliers dataset
train_size_outliers = int(len(y_outliers) * 0.8)
train_predict_lstm_outliers = predictions_lstm_outliers[:train_size_outliers]
test_predict_lstm_outliers = predictions_lstm_outliers[train_size_outliers:]

y_train_outliers_actual = y_outliers[:train_size_outliers]
y_test_outliers_actual = y_outliers[train_size_outliers:]

# Calculate RMSE for LSTM on outliers
train_rmse_lstm_outliers = np.sqrt(mean_squared_error(y_train_outliers_actual, train_predict_lstm_outliers))
test_rmse_lstm_outliers = np.sqrt(mean_squared_error(y_test_outliers_actual, test_predict_lstm_outliers))

# Print RMSE for LSTM model on outliers
print(f'LSTM Outliers Train RMSE: {train_rmse_lstm_outliers}')
print(f'LSTM Outliers Test RMSE: {test_rmse_lstm_outliers}')

from sklearn.metrics import mean_absolute_error

# Calculate MAE for each model on the whole dataset
train_mae_lgbm_whole = mean_absolute_error(y_train_whole, train_predict_lgbm_whole)
test_mae_lgbm_whole = mean_absolute_error(y_test_whole, test_predict_lgbm_whole)

train_mae_xgb_whole = mean_absolute_error(y_train_whole, train_predict_xgb_whole)
test_mae_xgb_whole = mean_absolute_error(y_test_whole, test_predict_xgb_whole)

# Calculate MAE for combined predictions
train_mae_combined = mean_absolute_error(y_train_whole, train_predict_combined)
test_mae_combined = mean_absolute_error(y_test_whole, test_predict_combined)

# Print the MAE values
print(f'LSTM+LightGBM+XGBoost Combined Train MAE: {train_mae_combined}')
print(f'LSTM+LightGBM+XGBoost Combined Test MAE: {test_mae_combined}')

print(f'LightGBM Train MAE: {train_mae_lgbm_whole}')
print(f'LightGBM Test MAE: {test_mae_lgbm_whole}')

print(f'XGBoost Train MAE: {train_mae_xgb_whole}')
print(f'XGBoost Test MAE: {test_mae_xgb_whole}')
