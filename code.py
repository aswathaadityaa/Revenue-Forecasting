import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, LSTM, Dense, Dropout
import os

np.random.seed(42)
tf.keras.utils.set_random_seed(42)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

FILE_PATH = "C:/Users/strai/Downloads/revenue.xlsx"
LOOK_BACK = 8
FORECAST_STEPS = 8
TRAIN_SPLIT = 0.8

try:
    data = pd.read_excel(FILE_PATH)
    REQUIRED = {'year', 'quarter', 'revenue'}
    if not REQUIRED.issubset(set(data.columns)):
        raise ValueError(f"File must contain columns: {sorted(REQUIRED)}")
    data = data.dropna(subset=['year', 'quarter', 'revenue']).copy()
    data[['year','quarter','revenue']] = data[['year','quarter','revenue']].apply(pd.to_numeric, errors='coerce')
    data = data.dropna(subset=['year','quarter','revenue'])
    data = data.sort_values(by=["year", "quarter"])
    data['date'] = pd.PeriodIndex.from_fields(year=data['year'], quarter=data['quarter'], freq='Q').to_timestamp(how='end')
    ts = data.set_index('date')['revenue'].asfreq('Q')
except FileNotFoundError:
    date_rng = pd.date_range(start='2014-01-01', end='2023-12-31', freq='Q')
    n_samples = len(date_rng)
    trend = np.linspace(100, 250, n_samples)
    seasonality = 15 * np.sin(np.arange(n_samples) * (np.pi / 2))
    noise = np.random.normal(0, 10, n_samples)
    ts = pd.Series(trend + seasonality + noise, index=date_rng)

train_size = int(len(ts) * TRAIN_SPLIT)
train, test = ts[0:train_size], ts[train_size:len(ts)]

arima_model = ARIMA(train, order=(1, 1, 1)).fit()
arima_residuals_train = arima_model.resid.dropna()
arima_test_forecast = arima_model.forecast(steps=len(test))

def create_dataset(series, look_back):
    X, y = [], []
    for i in range(len(series) - look_back):
        X.append(series[i:(i + look_back)])
        y.append(series[i + look_back])
    return np.array(X), np.array(y)

scaler = MinMaxScaler()
scaler.fit(arima_residuals_train.values.reshape(-1,1))
residuals_scaled = scaler.transform(arima_residuals_train.values.reshape(-1,1)).ravel()
X_train, y_train = create_dataset(residuals_scaled, LOOK_BACK)
X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

cnn_model = Sequential([
    Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(LOOK_BACK, 1)),
    Flatten(),
    Dense(50, activation='relu'),
    Dense(1)
])
cnn_model.compile(optimizer='adam', loss='mse')
cnn_model.fit(X_train_reshaped, y_train, epochs=200, verbose=0)

lstm_model = Sequential([
    LSTM(50, activation='relu', input_shape=(LOOK_BACK, 1)),
    Dropout(0.2),
    Dense(1)
])
lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.fit(X_train_reshaped, y_train, epochs=200, verbose=0)

def forecast_residuals(model, initial_sequence, steps):
    predicted_residuals = []
    current_sequence = initial_sequence.copy()
    for _ in range(steps):
        input_seq = current_sequence.reshape(1, LOOK_BACK, 1)
        pred_scaled = model.predict(input_seq, verbose=0)[0][0]
        predicted_residuals.append(pred_scaled)
        current_sequence = np.append(current_sequence[1:], pred_scaled)
    return scaler.inverse_transform(np.array(predicted_residuals).reshape(-1, 1)).flatten()

last_known_residuals_scaled = residuals_scaled[-LOOK_BACK:]
predicted_test_residuals_cnn = forecast_residuals(cnn_model, last_known_residuals_scaled, len(test))
predicted_test_residuals_lstm = forecast_residuals(lstm_model, last_known_residuals_scaled, len(test))

hybrid_cnn_test_forecast = arima_test_forecast.values + predicted_test_residuals_cnn
hybrid_lstm_test_forecast = arima_test_forecast.values + predicted_test_residuals_lstm

metrics = {
    'ARIMA': {
        'RMSE': np.sqrt(mean_squared_error(test, arima_test_forecast)),
        'MAE': mean_absolute_error(test, arima_test_forecast)
    },
    'ARIMA-CNN': {
        'RMSE': np.sqrt(mean_squared_error(test, hybrid_cnn_test_forecast)),
        'MAE': mean_absolute_error(test, hybrid_cnn_test_forecast)
    },
    'ARIMA-LSTM': {
        'RMSE': np.sqrt(mean_squared_error(test, hybrid_lstm_test_forecast)),
        'MAE': mean_absolute_error(test, hybrid_lstm_test_forecast)
    }
}

full_arima_model = ARIMA(ts, order=(1, 1, 1)).fit()
final_arima_forecast = full_arima_model.get_forecast(steps=FORECAST_STEPS)
forecast_index = final_arima_forecast.row_labels
final_arima_mean = final_arima_forecast.predicted_mean
full_arima_residuals = full_arima_model.resid.dropna()

full_residuals_scaled = scaler.fit_transform(full_arima_residuals.values.reshape(-1, 1)).flatten()
X_full, y_full = create_dataset(full_residuals_scaled, LOOK_BACK)
X_full_reshaped = X_full.reshape((X_full.shape[0], X_full.shape[1], 1))
cnn_model.fit(X_full_reshaped, y_full, epochs=200, verbose=0)
lstm_model.fit(X_full_reshaped, y_full, epochs=200, verbose=0)

last_full_residuals_scaled = full_residuals_scaled[-LOOK_BACK:]
final_predicted_residuals_cnn = forecast_residuals(cnn_model, last_full_residuals_scaled, FORECAST_STEPS)
final_predicted_residuals_lstm = forecast_residuals(lstm_model, last_full_residuals_scaled, FORECAST_STEPS)

final_hybrid_cnn_forecast = final_arima_mean.values + final_predicted_residuals_cnn
final_hybrid_lstm_forecast = final_arima_mean.values + final_predicted_residuals_lstm

plt.style.use('seaborn-v0_8-whitegrid')
fig = plt.figure(figsize=(18, 16))
gs = fig.add_gridspec(2, 1, height_ratios=[1, 2])
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

fig.suptitle('Comprehensive Revenue Forecasting Analysis', fontsize=22, weight='bold')

model_names = list(metrics.keys())
rmse_values = [m['RMSE'] for m in metrics.values()]
mae_values = [m['MAE'] for m in metrics.values()]

x = np.arange(len(model_names))
width = 0.35

rects1 = ax1.bar(x - width/2, rmse_values, width, label='RMSE', color='cornflowerblue')
rects2 = ax1.bar(x + width/2, mae_values, width, label='MAE', color='salmon')

ax1.set_ylabel('Error Value', fontsize=12)
ax1.set_title('Model Performance Comparison on Test Set', fontsize=16)
ax1.set_xticks(x)
ax1.set_xticklabels(model_names, fontsize=12)
ax1.legend()
ax1.bar_label(rects1, padding=3, fmt='%.2f')
ax1.bar_label(rects2, padding=3, fmt='%.2f')

ax2.plot(ts.index, ts, label='Historical Data', color='black', linewidth=2)
ax2.plot(forecast_index, final_arima_mean, label='ARIMA Forecast', color='blue', linestyle='--', marker='^')
ax2.plot(forecast_index, final_hybrid_cnn_forecast, label='ARIMA-CNN Forecast', color='red', linestyle='--', marker='s')
ax2.plot(forecast_index, final_hybrid_lstm_forecast, label='ARIMA-LSTM Forecast', color='green', linestyle='--', marker='o')
ax2.axvspan(ts.index[-1], forecast_index[-1], facecolor='gray', alpha=0.15, label='Forecast Period')
ax2.set_title(f'Final Forecast for Next {FORECAST_STEPS} Quarters', fontsize=16)
ax2.set_xlabel('Date', fontsize=12)
ax2.set_ylabel('Revenue', fontsize=12)
ax2.legend(loc='upper left', fontsize=11)
ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

final_forecast_df = pd.DataFrame({
    "Quarter": pd.PeriodIndex(forecast_index, freq='Q').astype(str),
    "ARIMA_Forecast": final_arima_mean.values,
    "ARIMA_CNN_Forecast": final_hybrid_cnn_forecast,
    "ARIMA_LSTM_Forecast": final_hybrid_lstm_forecast
})

output_csv_path = 'stovekraft_revenue_forecast.csv'
final_forecast_df.to_csv(output_csv_path, index=False)
print(final_forecast_df)
