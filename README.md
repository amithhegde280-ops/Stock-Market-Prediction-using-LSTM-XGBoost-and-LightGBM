Forecasting the Future: A Hybrid Approach to Stock Market Prediction
Overview

This project looks at stock price prediction using a hybrid machine learning approach. The idea is simple: use an LSTM network to handle the sudden spikes and outliers in stock data, then combine its output with boosting models like LightGBM and XGBoost to capture the overall trend.

The experiments were carried out on Google (GOOGL) stock data from 2004 to 2022. The hybrid approach ended up performing better than individual models, giving a more balanced view of both volatility and long-term movement.

Methodology
1. Data Preprocessing and Outlier Detection

Raw stock data was cleaned and converted into log returns.

Outliers (extreme highs and lows) were identified using the Z-score method.

These outliers were not discarded — instead, they were separated and later used to train the LSTM model.

2. LSTM on Outliers

An LSTM network was trained specifically on the outlier sequences.

The model had multiple LSTM layers followed by a dense output layer.

Training was done with the Adam optimizer and MSE loss.

Purpose: to capture short-term dependencies and extreme price moves.

3. Gradient Boosting Models on the Whole Dataset

LightGBM and XGBoost were trained on the complete dataset.

The predictions from the LSTM model were added as extra features.

Purpose: to use LSTM’s ability to capture sequence patterns while letting the boosting models handle accuracy and trend prediction.

4. Evaluation

Models were compared using RMSE and MAE.

Model	RMSE	MAE
LSTM + LightGBM	0.0637	0.0510
LSTM + CatBoost	0.0661	0.0529
LSTM + XGBoost	0.0674	0.0539
LSTM + LightGBM + XGBoost (Hybrid)	0.0653	0.0450

The hybrid model managed to capture both the general trend and sudden fluctuations better than standalone models.

Results

The hybrid (LSTM + LightGBM + XGBoost) achieved about 2.3% higher accuracy than the others.

LSTM helped handle outliers, while boosting models improved long-term accuracy.

The approach shows that combining models can strike a balance between trend following and volatility handling.

License

This project is licensed under All Rights Reserved.
You may view the code, but using, modifying, or redistributing it without permission is not allowed.
