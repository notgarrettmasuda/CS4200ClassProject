import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import math
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


TICKERS_TO_RUN = ["TSLA"]  #Ticker = Agilent Technologies
FOLDERS = ["stocks", "etfs"]

#Sequence Generator
def create_sequences(data, lookback=30):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i])
        y.append(data[i, 0])  # Predict 'Close'
    return np.array(X), np.array(y)

#MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

#Main Trainer
def train_on_file(filepath):
    print(f"\n📁 Processing: {filepath}")
    try:
        df = pd.read_csv(filepath)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

        if len(df) < 31:
            print(f"⛔ Skipped {filepath} (not enough rows)")
            return

        data = df[['Close', 'Volume']].values
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)

        X, y = create_sequences(scaled_data)
        if len(X) == 0:
            print(f"⛔ Skipped {filepath} (sequence creation failed)")
            return

        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model = Sequential()
        model.add(LSTM(units=50, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=0)

        predicted = model.predict(X_test)
        
        # Inverse transform predictions and actuals
        predicted_prices = scaler.inverse_transform(
            np.concatenate((predicted, X_test[:, -1, 1].reshape(-1, 1)), axis=1))[:, 0]
        actual_prices = scaler.inverse_transform(
            np.concatenate((y_test.reshape(-1, 1), X_test[:, -1, 1].reshape(-1, 1)), axis=1))[:, 0]

        # Accuracy Metrics
        rmse = math.sqrt(mean_squared_error(actual_prices, predicted_prices))
        mape = mean_absolute_percentage_error(actual_prices, predicted_prices)
        print(f"📉 RMSE for {filepath}: ${rmse:.2f}")
        print(f"📊 MAPE for {filepath}: {mape:.2f}%")

        # Save plot
        os.makedirs("plots", exist_ok=True)
        symbol = os.path.basename(filepath).replace(".csv", "")
        plot_path = f"plots/{symbol}_plot.png"

        plt.figure()
        plt.plot(actual_prices, label='Actual')
        plt.plot(predicted_prices, label='Predicted')
        plt.title(f"{symbol} Prediction\nMAPE: {mape:.2f}% | RMSE: ${rmse:.2f}")
        plt.xlabel("Days")
        plt.ylabel("Price ($)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        print(f"✅ Plot saved: {plot_path}")

    except Exception as e:
        print(f"❌ Error processing {filepath}: {e}")

#Run Models for Selected Ticker
for folder in FOLDERS:
    for ticker in TICKERS_TO_RUN:
        filepath = os.path.join(folder, f"{ticker}.csv")
        if os.path.exists(filepath):
            train_on_file(filepath)
